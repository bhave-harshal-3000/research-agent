"""
Deterministic reliability helpers for the research agent.

This module has no dependency on API keys or the network, so everything in it
can be unit-tested offline. It covers:

1. Evidence compression  : dedupe, relevance filtering, per-task caps, trimming.
2. Claim verification    : citation IDs, numeric checks, lexical support.
3. Confidence capping    : the model's confidence can only be lowered by code.
4. Robust LLM calls      : 413 -> shrink evidence, 429 -> backoff, bad JSON -> retry once.
"""
import re
import time
from datetime import datetime
from urllib.parse import urlparse

from pydantic import ValidationError

from schemas import ResearchReport, SourceItem

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
MIN_RELEVANCE = 0.25          # items scoring below this are dropped as off-topic
PER_TASK_CAP = 3              # max evidence items kept per research task
MAX_EVIDENCE = 14             # max evidence items passed to synthesis
SNIPPET_CHARS = 450           # max characters kept per snippet
TOKEN_BUDGET = 4000           # approx. token budget for the evidence block (Groq free tier: 8k TPM)
CLAIM_SUPPORT_MIN = 0.4       # min share of a claim's content words found in its cited sources
MAX_SHRINK_LEVELS = 3         # how many times a 413 may shrink the evidence
MAX_RATE_LIMIT_RETRIES = 3    # how many times a 429 may be retried with backoff
TRUSTED_BONUS = 0.1           # ranking boost for sources from recognised reliable domains
CLAIM_LABEL_SPAN = 2          # words either side of a number in a claim that name its metric
SOURCE_CONTEXT_SPAN = 3       # words either side of that number in the source where the metric must appear

INSUFFICIENT_ANSWER = "Insufficient verified evidence to answer this question reliably."


def ts() -> str:
    """Return formatted timestamp."""
    return datetime.now().strftime("%H:%M:%S")


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------
_STOPWORDS = frozenset("""
a an and are as at be been being but by can could did do does for from had has have how if in into is it its
may might more most not of on or our such than that the their them then there these they this those to was
were what when where which while who whom why will with would you your versus vs between among across about
after also any both each few other over same some very via per
analysis approach approaches based using use used study studies paper papers results result new method methods
recent latest compare compared comparison evaluate evaluation include including existing exist
""".split())


def content_tokens(text: str) -> set[str]:
    """Lowercased content words reduced to 5-char prefixes (cheap stemming: graph/graphs/graphrag match)."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w[:5] for w in words if len(w) > 2 and w not in _STOPWORDS and not w.isdigit()}


def trim_text(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "..."


def estimate_tokens(text: str) -> int:
    return len(text) // 4 + 1


_NUM_TOKEN_RE = re.compile(r"\d+(?:[.,]\d+)*%?")
_WORD_OR_NUM_RE = re.compile(r"\d+(?:[.,]\d+)*%?|[A-Za-z]+")
_CITATION_RE = re.compile(r"\[\d+(?:\s*,\s*\d+)*\]")


def _norm_num(token: str) -> str:
    return token.replace(",", "").rstrip("%")


def _is_significant(token: str) -> bool:
    """Bare single digits are ignored: 'two-stage', '3D', 'Task 1', ..."""
    return not (token.isdigit() and len(token) == 1)


def significant_numbers(text: str) -> set[str]:
    """Numbers worth checking, normalised ('11,720' -> '11720', '18%' -> '18')."""
    return {_norm_num(t) for t in _NUM_TOKEN_RE.findall(text) if _is_significant(t)}


def number_context_issues(claim: str, cited_text: str) -> list[str]:
    """
    Numbers in the claim that appear in the cited text, but never next to the metric the claim attaches
    them to. Catches a real number reused for the wrong statistic, e.g. a Test average of 28.40 cited as an
    ODI 'strike rate'. The metric is read from the words around the number in the claim and must appear
    within SOURCE_CONTEXT_SPAN words of the same number in the source. Heuristic: abbreviations such as
    'avg' for 'average' are not matched, so this errs on the side of rejecting.
    """
    claim_words = _WORD_OR_NUM_RE.findall(claim)
    cited_words = _WORD_OR_NUM_RE.findall(cited_text)

    positions: dict[str, list[int]] = {}
    for j, word in enumerate(cited_words):
        if word[0].isdigit():
            positions.setdefault(_norm_num(word), []).append(j)

    def labels_around(words: list[str], idx: int, span: int) -> set[str]:
        neighbours = words[max(0, idx - span):idx] + words[idx + 1:idx + 1 + span]
        return content_tokens(" ".join(w for w in neighbours if not w[0].isdigit()))

    issues: list[str] = []
    for i, word in enumerate(claim_words):
        if not word[0].isdigit() or not _is_significant(word):
            continue
        number = _norm_num(word)
        if number not in positions or number in issues:
            continue  # numbers missing from the source are reported separately
        labels = labels_around(claim_words, i, CLAIM_LABEL_SPAN)
        if labels and not any(labels & labels_around(cited_words, j, SOURCE_CONTEXT_SPAN) for j in positions[number]):
            issues.append(number)
    return issues


# ---------------------------------------------------------------------------
# 1. Evidence compression
# ---------------------------------------------------------------------------
def normalize_url(url: str) -> str:
    """Canonical key for dedupe; arXiv abs/html/pdf and versions collapse to one ID."""
    u = url.strip().lower()
    m = re.search(r"arxiv\.org/(?:abs|html|pdf)/(\d{4}\.\d{4,5})", u)
    if m:
        return f"arxiv:{m.group(1)}"
    u = re.sub(r"^https?://(www\.)?", "", u)
    return u.split("#")[0].rstrip("/")


def _has_url(item: dict) -> bool:
    url = (item.get("url") or "").strip()
    return bool(url) and url.upper() != "N/A"


# Social and user-generated sites: anyone can post numbers there, so they are never used as evidence.
# The same list is passed to Tavily as exclude_domains so these results do not waste result slots.
UNRELIABLE_DOMAINS = (
    "facebook.com", "fb.com", "fb.watch", "instagram.com", "tiktok.com", "x.com", "twitter.com",
    "youtube.com", "youtu.be", "reddit.com", "quora.com", "pinterest.com", "scribd.com",
    "slideshare.net", "linkedin.com", "threads.net",
)
# Recognised reliable domains (editable heuristic, not a ground truth). Everything else is 'unrated':
# still usable, but a report resting only on unrated sources cannot reach High confidence.
TRUSTED_DOMAINS = (
    # reference and scholarly
    "wikipedia.org", "arxiv.org", "doi.org", "nature.com", "science.org", "sciencedirect.com", "springer.com",
    "acm.org", "ieee.org", "aclanthology.org", "openreview.net", "mlr.press", "neurips.cc", "jstor.org",
    "semanticscholar.org", "pnas.org", "cell.com", "thelancet.com", "bmj.com", "nih.gov", "who.int",
    # news agencies and public broadcasters
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk",
    # official statistics databases
    "espncricinfo.com", "cricinfo.com", "cricbuzz.com", "howstat.com", "icc-cricket.com", "espn.com",
    "olympics.com", "fifa.com", "basketball-reference.com", "baseball-reference.com",
)


def domain_of(url: str) -> str:
    host = (urlparse(url if "://" in url else "http://" + url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def _domain_in(host: str, domains: tuple[str, ...]) -> bool:
    return any(host == d or host.endswith("." + d) for d in domains)


def source_tier(url: str) -> str:
    """'excluded' (social / user-generated), 'trusted' (recognised reliable domain) or 'unrated'."""
    host = domain_of(url)
    if _domain_in(host, UNRELIABLE_DOMAINS):
        return "excluded"
    padded = f".{host}."
    if (_domain_in(host, TRUSTED_DOMAINS) or host.endswith((".gov", ".edu", ".mil"))
            or any(f".{label}." in padded for label in ("ac", "gov", "edu"))):
        return "trusted"
    return "unrated"


def relevance_score(item: dict, question: str, task_text: str) -> float:
    """0..1 score: keyword overlap with the task and the question, blended with Tavily's own score."""
    item_tokens = content_tokens(f"{item.get('title', '')} {item.get('snippet', '')}")

    def overlap(reference: str) -> float:
        ref_tokens = content_tokens(reference)
        return len(ref_tokens & item_tokens) / len(ref_tokens) if ref_tokens else 0.0

    lexical = 0.5 * overlap(task_text) + 0.5 * overlap(question)
    engine_score = item.get("score")
    if isinstance(engine_score, (int, float)):
        return 0.5 * float(engine_score) + 0.5 * lexical
    return lexical


def _interleave_by_source(items: list[dict]) -> list[dict]:
    """Alternate source types (web / arXiv) so academic papers are not crowded out within a task."""
    groups: dict[str, list[dict]] = {}
    for item in items:
        groups.setdefault(item.get("source_type", ""), []).append(item)
    ordered = []
    while any(groups.values()):
        for group in groups.values():
            if group:
                ordered.append(group.pop(0))
    return ordered


def compress_evidence(raw: list[dict], question: str, max_items: int = MAX_EVIDENCE,
                      per_task_cap: int = PER_TASK_CAP,
                      snippet_chars: int = SNIPPET_CHARS) -> tuple[list[dict], dict]:
    """Dedupe, drop off-topic items, cap per task, trim snippets, and assign evidence IDs 1..N."""
    stats = {"raw": len(raw), "no_url": 0, "untrusted": 0, "duplicates": 0, "irrelevant": 0, "capped": 0, "kept": 0}
    seen: set[str] = set()
    scored: list[dict] = []

    for item in raw:
        if not _has_url(item):
            stats["no_url"] += 1
            continue
        tier = source_tier(item["url"])
        if tier == "excluded":
            stats["untrusted"] += 1
            print(f"   dropped (unreliable source {domain_of(item['url'])}): {trim_text(item.get('title', ''), 80)}")
            continue
        key = normalize_url(item["url"])
        if key in seen:
            stats["duplicates"] += 1
            continue
        seen.add(key)

        task_text = f"{item.get('task_subquestion', '')} {item.get('task_query', '')}"
        score = relevance_score(item, question, task_text)
        if score < MIN_RELEVANCE:
            stats["irrelevant"] += 1
            print(f"   dropped (irrelevant, score {score:.2f}): {trim_text(item.get('title', ''), 80)}")
            continue
        bonus = TRUSTED_BONUS if tier == "trusted" else 0.0
        scored.append({**item, "tier": tier, "relevance": round(score, 3), "priority": round(score + bonus, 3)})

    scored.sort(key=lambda i: i["priority"], reverse=True)

    by_task: dict[str, list[dict]] = {}
    for item in scored:
        by_task.setdefault(item.get("task_query", ""), []).append(item)
    queues = [_interleave_by_source(items)[:per_task_cap] for items in by_task.values()]

    # Round-robin across tasks so every task keeps its best evidence before any task gets a second item.
    selected: list[dict] = []
    for rank in range(per_task_cap):
        for queue in queues:
            if rank < len(queue) and len(selected) < max_items:
                selected.append(queue[rank])

    stats["capped"] = len(scored) - len(selected)
    stats["kept"] = len(selected)

    compressed = []
    for idx, item in enumerate(selected, 1):
        compressed.append({**item, "id": idx, "snippet": trim_text(item.get("snippet", ""), snippet_chars)})
    return compressed, stats


def shrink_evidence(items: list[dict], level: int) -> list[dict]:
    """Return a smaller view of the evidence (fewer items, shorter snippets) keeping the original IDs."""
    if level <= 0:
        return list(items)
    keep = min(len(items), max(3, len(items) // (2 ** level)))
    top = sorted(items, key=lambda e: e.get("priority", e.get("relevance", 0)), reverse=True)[:keep]
    top.sort(key=lambda e: e["id"])
    limit = max(120, int(SNIPPET_CHARS * (0.5 ** level)))
    return [{**e, "snippet": trim_text(e.get("snippet", ""), limit)} for e in top]


def evidence_to_text(items: list[dict]) -> str:
    blocks = [
        f"[{e['id']}] {e.get('source_type', 'Unknown')} | {e.get('title', 'Untitled')}\n"
        f"    URL: {e.get('url', '')} (source quality: {e.get('tier', 'unrated')})\n"
        f"    Snippet: {e.get('snippet', '')}"
        for e in items
    ]
    return "\n\n".join(blocks) if blocks else "No evidence available."


def initial_shrink_level(items: list[dict]) -> int:
    """Smallest shrink level whose evidence block fits TOKEN_BUDGET."""
    level = 0
    while level < MAX_SHRINK_LEVELS and estimate_tokens(evidence_to_text(shrink_evidence(items, level))) > TOKEN_BUDGET:
        level += 1
    return level


# ---------------------------------------------------------------------------
# 2. Claim verification
# ---------------------------------------------------------------------------
def verify_claim(claim: str, source_ids: list[int], by_id: dict[int, dict]) -> tuple[bool, str]:
    """
    Check one claim against the snippets it cites. Heuristic, not entailment:
      - it must cite at least one evidence ID, and every ID must exist;
      - every significant number in the claim must appear in the cited title/URL/snippet;
      - each number must sit next to the metric the claim attaches it to (see number_context_issues);
      - enough of the claim's content words must appear in the cited text.
    """
    if not source_ids:
        return False, "no source ids cited"
    unknown = sorted(set(source_ids) - set(by_id))
    if unknown:
        return False, f"unknown source id(s) {unknown}"

    cited_text = " ".join(
        f"{by_id[i].get('title', '')} {by_id[i].get('url', '')} {by_id[i].get('snippet', '')}" for i in source_ids
    )
    clean_claim = _CITATION_RE.sub("", claim)

    missing = significant_numbers(clean_claim) - significant_numbers(cited_text)
    if missing:
        return False, f"number(s) not found in cited sources: {', '.join(sorted(missing))}"
    misplaced = number_context_issues(clean_claim, cited_text)
    if misplaced:
        return False, f"number(s) not tied to the claimed metric in the cited text: {', '.join(misplaced)}"

    claim_tokens = content_tokens(clean_claim)
    if claim_tokens:
        support = len(claim_tokens & content_tokens(cited_text)) / len(claim_tokens)
        if support < CLAIM_SUPPORT_MIN:
            return False, f"low lexical support ({support:.2f} < {CLAIM_SUPPORT_MIN})"
    return True, "ok"


# ---------------------------------------------------------------------------
# 3. Confidence capping and report finalisation
# ---------------------------------------------------------------------------
_CONF_RANK = {"Low": 0, "Medium": 1, "High": 2}


def confidence_cap(n_verified: int, n_sources: int, n_dropped: int, n_trusted: int) -> tuple[str, str]:
    """Highest confidence the evidence can justify, with the reason when it is below High."""
    if n_verified == 0:
        return "Low", "no claim could be verified against its cited snippets"
    if n_verified < 2 or n_sources < 3:
        return "Low", f"only {n_verified} verified claim(s) from {n_sources} distinct source(s)"
    reasons = []
    if n_sources < 5:
        reasons.append(f"{n_sources} distinct verified source(s)")
    if n_dropped > n_verified:
        reasons.append(f"{n_dropped} dropped claim(s) against {n_verified} verified")
    if n_trusted == 0:
        reasons.append("none of the cited sources is from a recognised reliable domain")
    if reasons:
        return "Medium", "; ".join(reasons)
    return "High", ""


def insufficient_report(question: str, reason: str) -> ResearchReport:
    return ResearchReport(
        user_question=question,
        short_answer=INSUFFICIENT_ANSWER,
        key_findings=[],
        sources=[],
        confidence="Low",
        limitations_and_assumptions=[reason],
        suggested_next_steps=[
            "Check API keys, network access and tool errors in the log above.",
            "Rephrase the question or narrow it to a more specific topic.",
        ],
    )


def fallback_report(question: str, evidence: list[dict], reason: str) -> ResearchReport:
    """Used when synthesis cannot produce a valid answer. Lists retrieved sources but makes no claims."""
    return ResearchReport(
        user_question=question,
        short_answer="Synthesis failed, so no findings are reported. Retrieved sources are listed for manual review.",
        key_findings=[],
        sources=[SourceItem(id=e.get("id"), title=e.get("title", "Source"), url=e["url"])
                 for e in evidence if _has_url(e)],
        confidence="Low",
        limitations_and_assumptions=[f"Synthesis failed: {reason}", "Listed sources were retrieved but not analysed."],
        suggested_next_steps=["Re-run the agent; if the error was a rate limit, wait a minute first."],
    )


def find_coverage_gaps(tasks: list[dict], evidence: list[dict]) -> list[str]:
    """Subquestions of the plan for which no usable evidence survived curation."""
    covered = {e.get("task_query") for e in evidence}
    return [t.get("subquestion") or t.get("query", "") for t in tasks
            if t.get("query", t.get("subquestion")) not in covered]


def verify_and_finalize(draft: ResearchReport, evidence_view: list[dict], question: str,
                        coverage_gaps: list[str] | None = None) -> ResearchReport:
    """Verify every finding, rebuild sources from verified citations, and cap confidence."""
    by_id = {e["id"]: e for e in evidence_view}
    verified, dropped = [], []

    print(f"\nVERIFYING {len(draft.key_findings)} claim(s) against their cited snippets:")
    for finding in draft.key_findings:
        ok, reason = verify_claim(finding.claim, finding.source_ids, by_id)
        print(f"   [{'KEPT   ' if ok else 'DROPPED'}] {trim_text(finding.claim, 110)} "
              f"(ids={finding.source_ids}) -> {reason}")
        (verified if ok else dropped).append(finding)

    cited_ids = sorted({i for f in verified for i in f.source_ids})
    sources = [SourceItem(id=i, title=by_id[i].get("title", "Source"), url=by_id[i]["url"]) for i in cited_ids]

    notes: list[str] = []
    n_trusted = sum(1 for i in cited_ids if by_id[i].get("tier") == "trusted")
    caps: list[tuple[str, str]] = [confidence_cap(len(verified), len(cited_ids), len(dropped), n_trusted)]
    short_answer = draft.short_answer

    if not verified:
        short_answer = INSUFFICIENT_ANSWER
        notes.append("The model's short answer was replaced because none of its claims could be verified.")
    else:
        allowed_text = " ".join(
            [question]
            + [f"{e.get('title', '')} {e.get('url', '')} {e.get('snippet', '')}" for e in evidence_view]
            + [f.claim for f in verified]
        )
        unsupported = significant_numbers(short_answer) - significant_numbers(allowed_text)
        if unsupported:
            figures = ", ".join(sorted(unsupported))
            notes.append(f"The short answer contains figures not found in the evidence ({figures}); treat them as unverified.")
            caps.append(("Low", "short answer contains unsupported figures"))

    if dropped:
        notes.append(f"{len(dropped)} model-generated claim(s) were removed because they could not be verified "
                     f"against the cited snippets.")

    gaps = list(coverage_gaps or [])
    if gaps:
        shown = "; ".join(trim_text(g, 90) for g in gaps[:5])
        notes.append(f"No usable evidence was found for {len(gaps)} planned subquestion(s), so they remain unanswered: {shown}")
        caps.append(("Medium", f"{len(gaps)} planned subquestion(s) have no usable evidence"))

    final_conf, why = min(caps, key=lambda c: _CONF_RANK[c[0]])
    if _CONF_RANK[draft.confidence] > _CONF_RANK[final_conf]:
        notes.append(f"Confidence lowered from {draft.confidence} to {final_conf} by automated checks: {why}.")
    else:
        final_conf = draft.confidence

    print(f"   Verified: {len(verified)} | Dropped: {len(dropped)} | Distinct sources: {len(cited_ids)} | "
          f"Confidence: {draft.confidence} -> {final_conf}")

    next_steps = list(draft.suggested_next_steps)
    if not next_steps:
        next_steps = [f"Search specifically for: {trim_text(g, 120)}" for g in gaps[:3]]
        next_steps = next_steps or ["Check the key findings against the primary sources listed above."]

    return ResearchReport(
        user_question=question,
        short_answer=short_answer,
        key_findings=verified,
        sources=sources,
        confidence=final_conf,
        limitations_and_assumptions=list(draft.limitations_and_assumptions) + notes,
        suggested_next_steps=next_steps,
    )


# ---------------------------------------------------------------------------
# 4. Robust LLM calls
# ---------------------------------------------------------------------------
class LLMCallError(RuntimeError):
    """Raised when an LLM call cannot be completed or validated after all recovery attempts."""


def error_status(exc: Exception) -> int | None:
    """HTTP status of an API error (Groq errors expose status_code; fall back to the message text)."""
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status
    match = re.search(r"Error code:\s*(\d{3})", str(exc))
    return int(match.group(1)) if match else None


def retry_after_seconds(exc: Exception, attempt: int) -> float:
    """Wait suggested by the API ('try again in 1m3.5s' / '250ms') plus 1s margin, else exponential backoff."""
    match = re.search(r"try again in\s*(?:(\d+)m)?\s*([\d.]+)(ms|s)", str(exc), re.IGNORECASE)
    if match:
        minutes = int(match.group(1) or 0)
        value = float(match.group(2))
        seconds = minutes * 60 + (value / 1000 if match.group(3).lower() == "ms" else value)
        return min(seconds + 1.0, 65.0)
    return float(min(2 ** attempt, 30))


def invoke_validated(llm, label: str, build_messages, schema, *, start_level: int = 0,
                     max_shrink: int = MAX_SHRINK_LEVELS,
                     max_rate_retries: int = MAX_RATE_LIMIT_RETRIES,
                     sleep=time.sleep):
    """
    Call a JSON-mode LLM and validate the reply with a Pydantic schema.

      413 (request too large) -> rebuild the prompt one shrink level smaller (up to max_shrink times)
      429 (rate limited)      -> wait (API hint or exponential backoff) and retry (up to max_rate_retries)
      invalid JSON / schema   -> retry once with a corrective prompt
      anything else           -> LLMCallError immediately

    build_messages(level, corrective) must return the message list; `corrective` is None or the
    validation error text. Returns (parsed_object, shrink_level_used).
    """
    level = start_level
    max_level = start_level + max_shrink
    corrective = None
    validation_retried = False
    rate_retries = 0

    while True:
        messages = build_messages(level, corrective)
        started = time.perf_counter()
        print(f"Calling {label} LLM (shrink level {level}) at {ts()}...")

        try:
            response = llm.invoke(messages)
        except Exception as exc:  # API / network error
            elapsed = time.perf_counter() - started
            status = error_status(exc)
            if status == 413:
                if level < max_level:
                    level += 1
                    print(f"⚠️ {label}: request too large (413) after {elapsed:.2f}s. "
                          f"Shrinking evidence to level {level} and retrying.")
                    continue
                raise LLMCallError(f"{label}: request still too large after {max_shrink} shrink step(s)") from exc
            if status == 429:
                if rate_retries < max_rate_retries:
                    rate_retries += 1
                    wait = retry_after_seconds(exc, rate_retries)
                    print(f"⚠️ {label}: rate limited (429). Waiting {wait:.1f}s "
                          f"(retry {rate_retries}/{max_rate_retries})...")
                    sleep(wait)
                    continue
                raise LLMCallError(f"{label}: still rate limited after {max_rate_retries} retries") from exc
            raise LLMCallError(f"{label}: API error: {exc}") from exc

        elapsed = time.perf_counter() - started
        print(f"⏱️ {label} LLM responded in {elapsed:.2f}s")

        try:
            parsed = schema.model_validate_json(response.content)
        except ValidationError as exc:
            if not validation_retried:
                validation_retried = True
                corrective = trim_text(str(exc), 500)
                print(f"⚠️ {label}: validation failure: {corrective}")
                print(f"Retrying {label} once with corrective prompt...")
                continue
            raise LLMCallError(f"{label}: invalid output after one corrective retry: {exc}") from exc

        print(f"✅ {label} output validated with Pydantic.")
        return parsed, level

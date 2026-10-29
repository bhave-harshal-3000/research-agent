"""Offline evaluation tests for evidence curation, claim verification, confidence and LLM-call recovery."""
import json

import pytest

import reliability as rel
from schemas import Finding, ResearchPlan, ResearchReport

QUESTION = (
    "Compare GraphRAG, vector-based RAG, and knowledge-graph memory for long-term memory in AI agents. "
    "Evaluate retrieval accuracy, multi-hop reasoning, computational cost and scalability."
)
GOOD_SNIPPET = ("GraphRAG {w} improves multi-hop reasoning and retrieval accuracy over vector RAG "
                "for knowledge graph memory in AI agents.")


def make_item(title="Paper", url="https://example.com/a", snippet=None, source_type="Web (Tavily)",
              task_query="GraphRAG multi-hop reasoning evaluation",
              subq="How does GraphRAG perform multi-hop reasoning?", score=None):
    return {
        "source_type": source_type, "title": title, "url": url,
        "snippet": GOOD_SNIPPET.format(w=title) if snippet is None else snippet,
        "task_subquestion": subq, "task_query": task_query, "score": score,
    }


def numbered(items):
    return [{**item, "id": i} for i, item in enumerate(items, 1)]


# ----------------------------------------------------------------------------
# Evidence curation
# ----------------------------------------------------------------------------
def test_arxiv_html_abs_and_versions_are_deduplicated():
    raw = [
        make_item("A", "https://arxiv.org/html/2506.05690v3"),
        make_item("B", "http://arxiv.org/abs/2506.05690v2", source_type="Academic (arXiv)"),
        make_item("C", "https://example.com/other"),
    ]
    kept, stats = rel.compress_evidence(raw, QUESTION)
    assert stats["duplicates"] == 1
    assert len(kept) == 2


def test_trailing_slash_and_scheme_are_deduplicated():
    raw = [make_item("A", "https://www.example.com/post/"), make_item("B", "http://example.com/post")]
    kept, stats = rel.compress_evidence(raw, QUESTION)
    assert stats["duplicates"] == 1 and len(kept) == 1


def test_off_topic_papers_are_filtered():
    off_topic = make_item(
        "Hybrid monolithic and staggered solution techniques for fracture", "http://arxiv.org/abs/2203.15924v2",
        snippet="Finite element fracture mechanics of fibrous network materials, computational analysis.",
        source_type="Academic (arXiv)", task_query="GraphRAG computational cost scalability analysis",
        subq="What are the computational cost and scalability trade-offs for GraphRAG?")
    on_topic = make_item("alpha", "https://example.com/alpha")
    kept, stats = rel.compress_evidence([off_topic, on_topic], QUESTION)
    assert stats["irrelevant"] == 1
    assert [e["title"] for e in kept] == ["alpha"]


def test_items_without_usable_url_are_dropped():
    raw = [make_item("A", "N/A"), make_item("B", ""), make_item("C", "https://example.com/c")]
    kept, stats = rel.compress_evidence(raw, QUESTION)
    assert stats["no_url"] == 2 and len(kept) == 1


def test_caps_ids_and_snippet_length():
    raw = []
    for t in range(6):
        for i in range(5):
            raw.append(make_item(f"w{t}x{i}", f"https://example.com/{t}/{i}", snippet=GOOD_SNIPPET.format(w=f"w{t}x{i}") * 10,
                                 task_query=f"GraphRAG multi-hop reasoning topic{t}"))
    kept, _ = rel.compress_evidence(raw, QUESTION)
    assert len(kept) <= rel.MAX_EVIDENCE
    assert [e["id"] for e in kept] == list(range(1, len(kept) + 1))
    assert all(len(e["snippet"]) <= rel.SNIPPET_CHARS + 3 for e in kept)
    per_task = {}
    for e in kept:
        per_task[e["task_query"]] = per_task.get(e["task_query"], 0) + 1
    assert max(per_task.values()) <= rel.PER_TASK_CAP
    assert len(per_task) == 6, "every task should keep some evidence"


def test_academic_papers_are_not_crowded_out_by_web_results():
    raw = [make_item(f"web{i}", f"https://example.com/web{i}", score=0.95) for i in range(4)]
    raw += [make_item(f"paper{i}", f"http://arxiv.org/abs/2501.0000{i}v1", source_type="Academic (arXiv)") for i in range(3)]
    kept, _ = rel.compress_evidence(raw, QUESTION)
    assert any(e["source_type"] == "Academic (arXiv)" for e in kept)


def test_compressed_evidence_fits_token_budget():
    raw = [make_item(f"p{i}", f"https://example.com/{i}", snippet=GOOD_SNIPPET.format(w=f"p{i}") * 30,
                     task_query=f"GraphRAG multi-hop reasoning topic{i % 5}") for i in range(40)]
    kept, _ = rel.compress_evidence(raw, QUESTION)
    level = rel.initial_shrink_level(kept)
    assert rel.estimate_tokens(rel.evidence_to_text(rel.shrink_evidence(kept, level))) <= rel.TOKEN_BUDGET


def test_shrink_keeps_ids_and_reduces_size():
    items = numbered([make_item(f"p{i}", f"https://example.com/{i}") for i in range(12)])
    for i in items:
        i["relevance"] = 0.5
    small = rel.shrink_evidence(items, 2)
    assert len(small) < len(items)
    assert {e["id"] for e in small} <= {e["id"] for e in items}


# ----------------------------------------------------------------------------
# Claim verification
# ----------------------------------------------------------------------------
@pytest.fixture
def by_id():
    return {e["id"]: e for e in numbered([
        make_item("alpha", "https://example.com/alpha", snippet="GraphRAG improves multi-hop accuracy by 18% over vector RAG."),
        make_item("beta", "https://example.com/beta", snippet="GraphRAG indexing is expensive because building the graph needs many LLM calls."),
    ])}


def test_supported_claim_passes(by_id):
    ok, _ = rel.verify_claim("GraphRAG improves multi-hop accuracy over vector RAG", [1], by_id)
    assert ok


def test_number_present_in_snippet_passes(by_id):
    ok, _ = rel.verify_claim("GraphRAG improves multi-hop accuracy by 18% over vector RAG", [1], by_id)
    assert ok


def test_fabricated_number_is_rejected(by_id):
    ok, reason = rel.verify_claim("GraphRAG achieves 15-20% higher accuracy than vector RAG", [1], by_id)
    assert not ok and "number" in reason


def test_unknown_source_id_is_rejected(by_id):
    ok, reason = rel.verify_claim("GraphRAG improves multi-hop accuracy", [99], by_id)
    assert not ok and "unknown" in reason


def test_missing_citation_is_rejected(by_id):
    ok, _ = rel.verify_claim("GraphRAG improves multi-hop accuracy", [], by_id)
    assert not ok


def test_valid_id_does_not_prove_an_unrelated_claim(by_id):
    ok, reason = rel.verify_claim("Rotary positional embeddings stabilise transformer tokenization", [1], by_id)
    assert not ok and "lexical" in reason


# ----------------------------------------------------------------------------
# Report finalisation and confidence
# ----------------------------------------------------------------------------
WORDS = ["alpha", "beta", "gamma", "delta", "epsilon"]


def evidence_view(n=5):
    items = numbered([make_item(w, f"https://example.com/{w}") for w in WORDS[:n]])
    return [{**e, "tier": "trusted"} for e in items]


def draft_with(n_good, n_bad=0, confidence="High", short="GraphRAG improves multi-hop retrieval."):
    view = evidence_view(5)
    findings = [Finding(claim=GOOD_SNIPPET.format(w=WORDS[i]), source_ids=[i + 1]) for i in range(n_good)]
    findings += [Finding(claim="Totally unsupported 77% improvement on SomeBench", source_ids=[1]) for _ in range(n_bad)]
    return view, ResearchReport(user_question=QUESTION, short_answer=short, key_findings=findings, confidence=confidence)


def test_no_verified_claims_gives_low_confidence_and_insufficient_answer():
    view, draft = draft_with(0, n_bad=2)
    report = rel.verify_and_finalize(draft, view, QUESTION)
    assert report.confidence == "Low"
    assert report.short_answer == rel.INSUFFICIENT_ANSWER
    assert report.key_findings == [] and report.sources == []


def test_two_sources_cannot_be_high_confidence():
    view, draft = draft_with(2)
    assert rel.verify_and_finalize(draft, view, QUESTION).confidence == "Low"


def test_three_sources_capped_at_medium():
    view, draft = draft_with(3)
    assert rel.verify_and_finalize(draft, view, QUESTION).confidence == "Medium"


def test_five_verified_sources_keep_high_confidence():
    view, draft = draft_with(5)
    assert rel.verify_and_finalize(draft, view, QUESTION).confidence == "High"


def test_model_low_confidence_is_never_raised():
    view, draft = draft_with(5, confidence="Low")
    assert rel.verify_and_finalize(draft, view, QUESTION).confidence == "Low"


def test_dropped_claims_are_removed_and_reported():
    view, draft = draft_with(3, n_bad=1)
    report = rel.verify_and_finalize(draft, view, QUESTION)
    assert len(report.key_findings) == 3
    assert any("removed" in note for note in report.limitations_and_assumptions)


def test_short_answer_with_unsupported_figures_forces_low():
    view, draft = draft_with(5, short="GraphRAG is 42% better than vector RAG.")
    report = rel.verify_and_finalize(draft, view, QUESTION)
    assert report.confidence == "Low"
    assert any("figures" in note for note in report.limitations_and_assumptions)


def test_sources_are_built_from_verified_citations_and_exist_in_evidence():
    view, draft = draft_with(3, n_bad=1)
    report = rel.verify_and_finalize(draft, view, QUESTION)
    evidence_urls = {e["url"] for e in view}
    assert {s.url for s in report.sources} <= evidence_urls
    assert {s.id for s in report.sources} == {1, 2, 3}


def test_model_supplied_fake_sources_are_discarded():
    view, draft = draft_with(3)
    draft.sources = [rel.SourceItem(title="Fake", url="https://fake.example/made-up")]
    report = rel.verify_and_finalize(draft, view, QUESTION)
    assert all(s.url != "https://fake.example/made-up" for s in report.sources)


def test_confidence_is_case_insensitive_in_schema():
    assert ResearchReport(user_question="q", short_answer="a", confidence="medium").confidence == "Medium"


# ----------------------------------------------------------------------------
# LLM call recovery (413 shrink, 429 backoff, validation retry)
# ----------------------------------------------------------------------------
class StatusError(Exception):
    def __init__(self, status, message=""):
        super().__init__(f"Error code: {status} - {message}")
        self.status_code = status


class FakeResponse:
    def __init__(self, content):
        self.content = content


class ScriptedLLM:
    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return FakeResponse(step)


VALID_PLAN = json.dumps({"objective": "x", "tasks": [], "reasoning": ""})


def run(llm, **kwargs):
    seen = []

    def build(level, corrective):
        seen.append((level, corrective))
        return ["msg"]

    sleeps = []
    result = rel.invoke_validated(llm, "Test", build, ResearchPlan, sleep=sleeps.append, **kwargs)
    return result, seen, sleeps


def test_413_shrinks_evidence_then_succeeds():
    llm = ScriptedLLM([StatusError(413, "Request too large"), VALID_PLAN])
    (plan, level), seen, _ = run(llm)
    assert [lvl for lvl, _ in seen] == [0, 1] and level == 1 and plan.objective == "x"


def test_413_gives_up_after_max_shrink():
    llm = ScriptedLLM([StatusError(413)] * 3)
    with pytest.raises(rel.LLMCallError):
        run(llm, max_shrink=2)
    assert llm.calls == 3


def test_429_backs_off_using_api_hint_then_succeeds():
    llm = ScriptedLLM([StatusError(429, "Please try again in 2.5s."), StatusError(429, "try again in 250ms"), VALID_PLAN])
    (_, level), _, sleeps = run(llm)
    assert level == 0
    assert sleeps == [pytest.approx(3.5), pytest.approx(1.25)]


def test_429_gives_up_after_max_retries():
    llm = ScriptedLLM([StatusError(429)] * 3)
    with pytest.raises(rel.LLMCallError):
        run(llm, max_rate_retries=2)
    assert llm.calls == 3


def test_retry_after_parses_minutes_and_falls_back_to_exponential():
    assert rel.retry_after_seconds(StatusError(429, "try again in 1m3.5s"), 1) == pytest.approx(64.5)
    assert rel.retry_after_seconds(StatusError(429, "no hint"), 1) == 2.0
    assert rel.retry_after_seconds(StatusError(429, "no hint"), 3) == 8.0


def test_invalid_json_is_retried_once_with_corrective_prompt():
    llm = ScriptedLLM(["not json at all", VALID_PLAN])
    (plan, _), seen, _ = run(llm)
    assert llm.calls == 2 and seen[0][1] is None and seen[1][1]


def test_invalid_json_twice_raises_instead_of_looping():
    llm = ScriptedLLM(["bad", "still bad"])
    with pytest.raises(rel.LLMCallError):
        run(llm)
    assert llm.calls == 2


def test_other_api_errors_are_not_retried():
    llm = ScriptedLLM([StatusError(500, "boom")])
    with pytest.raises(rel.LLMCallError):
        run(llm)
    assert llm.calls == 1


# ----------------------------------------------------------------------------
# Regression tests for the cricket-stats failure: unreliable sources, numbers attached to the wrong
# metric, and planned subquestions that silently returned nothing.
# ----------------------------------------------------------------------------
def test_source_tiers():
    assert rel.source_tier("https://en.wikipedia.org/wiki/Travis_Head") == "trusted"
    assert rel.source_tier("http://arxiv.org/abs/2506.05690v3") == "trusted"
    assert rel.source_tier("https://www.espncricinfo.com/player/x") == "trusted"
    assert rel.source_tier("https://www.ox.ac.uk/research") == "trusted"
    assert rel.source_tier("https://www.facebook.com/dreamcomparisonn/videos/1") == "excluded"
    assert rel.source_tier("https://m.youtube.com/watch?v=1") == "excluded"
    assert rel.source_tier("https://someblog.example.com/post") == "unrated"


def test_social_media_is_dropped_and_trusted_sources_rank_first():
    raw = [
        make_item("fb", "https://www.facebook.com/x/videos/1"),
        make_item("yt", "https://www.youtube.com/watch?v=abc"),
        make_item("blog", "https://someblog.example.com/post"),
        make_item("wiki", "https://en.wikipedia.org/wiki/GraphRAG"),
    ]
    kept, stats = rel.compress_evidence(raw, QUESTION)
    assert stats["untrusted"] == 2
    assert [e["title"] for e in kept] == ["wiki", "blog"]


STATS_SNIPPET = ("Travis Head Test batting average 28.40 with two Test hundreds. "
                 "In ODIs his strike rate is 100.5 and he has 7 centuries.")


def stats_source(snippet):
    return {1: {"id": 1, "title": "Player", "url": "https://en.wikipedia.org/wiki/Player", "snippet": snippet}}


def test_real_number_attached_to_the_wrong_metric_is_rejected():
    ok, reason = rel.verify_claim("Travis Head has a 28.40 ODI strike rate", [1], stats_source(STATS_SNIPPET))
    assert not ok and "metric" in reason
    ok, _ = rel.verify_claim("Travis Head has an ODI strike rate of 100.5", [1], stats_source(STATS_SNIPPET))
    assert ok


def test_correct_numbers_with_their_metrics_pass():
    source = stats_source("Rohit Sharma ODI batting average 48.83 with 31 centuries.")
    ok, _ = rel.verify_claim("Rohit Sharma has an ODI batting average of 48.83 and 31 centuries", [1], source)
    assert ok


def test_confidence_is_capped_at_medium_without_a_trusted_source():
    view = [{**e, "tier": "unrated"} for e in evidence_view(5)]
    _, draft = draft_with(5)
    assert rel.verify_and_finalize(draft, view, QUESTION).confidence == "Medium"


def test_unanswered_subquestions_are_reported_and_cap_confidence():
    view, draft = draft_with(5)
    report = rel.verify_and_finalize(draft, view, QUESTION, coverage_gaps=["Virat Kohli ODI career statistics"])
    assert report.confidence == "Medium"
    assert any("Virat Kohli" in note for note in report.limitations_and_assumptions)
    assert "Kohli" in report.suggested_next_steps[0]


def test_find_coverage_gaps():
    tasks = [{"subquestion": "A?", "query": "qa"}, {"subquestion": "B?", "query": "qb"}]
    assert rel.find_coverage_gaps(tasks, [{"task_query": "qa"}]) == ["B?"]

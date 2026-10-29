"""Offline end-to-end tests of the LangGraph nodes using fake tools and fake LLMs."""
import json

import pytest

import agent
import reliability as rel

QUESTION = (
    "Compare GraphRAG, vector-based RAG, and knowledge-graph memory for long-term memory in AI agents. "
    "Evaluate retrieval accuracy, multi-hop reasoning, computational cost and scalability."
)
SNIPPET = ("GraphRAG {w} improves multi-hop reasoning and retrieval accuracy over vector RAG "
           "for knowledge graph memory in AI agents.")
PLAN = {
    "objective": "compare",
    "tasks": [{"subquestion": "How does GraphRAG perform multi-hop reasoning?",
               "tool": "both", "query": "GraphRAG multi-hop reasoning evaluation"}],
    "reasoning": "",
}


class StatusError(Exception):
    def __init__(self, status, message=""):
        super().__init__(f"Error code: {status} - {message}")
        self.status_code = status


class ScriptedLLM:
    """Pops one scripted step per call; records the size of each prompt."""
    def __init__(self, script):
        self.script = list(script)
        self.prompt_sizes = []

    def invoke(self, messages):
        self.prompt_sizes.append(sum(len(m.content) for m in messages))
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step

        class Response:
            content = step
        return Response()


class ForbiddenLLM:
    def invoke(self, messages):
        raise AssertionError("The LLM must not be called in this scenario")


def tool_item(word, source_type="Web (Tavily)", url=None):
    return {"source_type": source_type, "title": f"Paper {word}", "url": url or f"https://example.com/{word}",
            "snippet": SNIPPET.format(w=word), "task_subquestion": PLAN["tasks"][0]["subquestion"],
            "task_query": PLAN["tasks"][0]["query"], "score": None}


def curated(words):
    evidence, _ = rel.compress_evidence([tool_item(w, url=f"https://example.com/{w}") for w in words], QUESTION)
    return evidence


def draft_json(findings, confidence="High", short="GraphRAG improves multi-hop retrieval."):
    return json.dumps({
        "user_question": QUESTION, "short_answer": short, "key_findings": findings,
        "confidence": confidence, "limitations_and_assumptions": [], "suggested_next_steps": [],
    })


def id_of(evidence, word):
    return next(e["id"] for e in evidence if word in e["title"])


# ----------------------------------------------------------------------------
# Researcher
# ----------------------------------------------------------------------------
def test_tavily_failure_is_swallowed_and_logged(monkeypatch):
    class Boom:
        def invoke(self, _):
            raise RuntimeError("network down")
    monkeypatch.setattr(agent, "tavily", Boom())
    assert agent.run_tavily_search("anything") == []


def test_tavily_results_keep_their_relevance_score(monkeypatch):
    class Fake:
        def invoke(self, _):
            return {"results": [{"title": "T", "url": "https://example.com", "content": "c", "score": 0.8}]}
    monkeypatch.setattr(agent, "tavily", Fake())
    assert agent.run_tavily_search("q")[0]["score"] == 0.8


def test_researcher_keeps_arxiv_results_when_tavily_fails(monkeypatch):
    monkeypatch.setattr(agent, "run_tavily_search", lambda q: [])  # simulates a swallowed Tavily failure
    monkeypatch.setattr(agent, "run_arxiv_search",
                        lambda q, max_results=4: [tool_item("alpha", "Academic (arXiv)", "http://arxiv.org/abs/2501.00001v1")])
    result = agent.researcher({"question": QUESTION, "plan": PLAN})
    assert len(result["evidence"]) == 1 and result["evidence"][0]["id"] == 1


def test_researcher_with_all_tools_failing_returns_empty_evidence(monkeypatch):
    monkeypatch.setattr(agent, "run_tavily_search", lambda q: [])
    monkeypatch.setattr(agent, "run_arxiv_search", lambda q, max_results=4: [])
    assert agent.researcher({"question": QUESTION, "plan": PLAN})["evidence"] == []


def test_researcher_with_no_tasks_returns_empty_evidence():
    assert agent.researcher({"question": QUESTION, "plan": {"tasks": []}})["evidence"] == []


# ----------------------------------------------------------------------------
# Synthesis
# ----------------------------------------------------------------------------
def test_empty_evidence_returns_insufficient_report_without_calling_llm(monkeypatch):
    monkeypatch.setattr(agent, "json_llm", ForbiddenLLM())
    report = agent.synthesize({"question": QUESTION, "plan": PLAN, "evidence": []})["report"]
    assert report["confidence"] == "Low"
    assert report["short_answer"] == rel.INSUFFICIENT_ANSWER
    assert report["sources"] == [] and report["key_findings"] == []


def test_hallucinated_numbers_and_invalid_ids_are_dropped(monkeypatch):
    evidence = curated(["alpha", "beta", "gamma"])
    a = id_of(evidence, "alpha")
    findings = [
        {"claim": SNIPPET.format(w="alpha"), "source_ids": [a]},
        {"claim": "GraphRAG delivers 15-20% higher accuracy on OpenBookQA and HotpotQA", "source_ids": [a]},
        {"claim": SNIPPET.format(w="beta"), "source_ids": [42]},
    ]
    monkeypatch.setattr(agent, "json_llm", ScriptedLLM([draft_json(findings)]))
    report = agent.synthesize({"question": QUESTION, "plan": PLAN, "evidence": evidence})["report"]
    assert len(report["key_findings"]) == 1
    assert report["confidence"] == "Low"
    assert "OpenBookQA" not in json.dumps(report)


def test_every_reported_source_comes_from_retrieved_evidence(monkeypatch):
    evidence = curated(["alpha", "beta", "gamma"])
    findings = [{"claim": SNIPPET.format(w=w), "source_ids": [id_of(evidence, w)]} for w in ("alpha", "beta", "gamma")]
    monkeypatch.setattr(agent, "json_llm", ScriptedLLM([draft_json(findings)]))
    report = agent.synthesize({"question": QUESTION, "plan": PLAN, "evidence": evidence})["report"]
    assert report["sources"]
    assert {s["url"] for s in report["sources"]} <= {e["url"] for e in evidence}


def test_synthesis_recovers_from_413_with_a_smaller_prompt(monkeypatch):
    raw = [tool_item(f"w{i}", url=f"https://example.com/w{i}") for i in range(10)]
    for i, item in enumerate(raw):
        item["task_query"] = f"GraphRAG multi-hop reasoning topic{i % 4}"
    evidence, _ = rel.compress_evidence(raw, QUESTION)
    a = evidence[0]
    llm = ScriptedLLM([StatusError(413, "Request too large"),
                       draft_json([{"claim": a["snippet"], "source_ids": [a["id"]]}])])
    monkeypatch.setattr(agent, "json_llm", llm)
    report = agent.synthesize({"question": QUESTION, "plan": PLAN, "evidence": evidence})["report"]
    assert len(llm.prompt_sizes) == 2 and llm.prompt_sizes[1] < llm.prompt_sizes[0]
    assert report["short_answer"] != "Synthesis failed, so no findings are reported. Retrieved sources are listed for manual review."


def test_synthesis_api_failure_returns_safe_fallback(monkeypatch):
    evidence = curated(["alpha", "beta"])
    monkeypatch.setattr(agent, "json_llm", ScriptedLLM([StatusError(500, "server error")]))
    report = agent.synthesize({"question": QUESTION, "plan": PLAN, "evidence": evidence})["report"]
    assert report["confidence"] == "Low" and report["key_findings"] == []
    assert {s["url"] for s in report["sources"]} == {e["url"] for e in evidence}


def test_planner_falls_back_when_llm_fails(monkeypatch):
    monkeypatch.setattr(agent, "json_llm", ScriptedLLM([StatusError(500, "boom")]))
    plan = agent.planner({"question": QUESTION})["plan"]
    assert len(plan["tasks"]) == 1 and plan["tasks"][0]["tool"] == "both"


def test_planner_accepts_valid_plan(monkeypatch):
    monkeypatch.setattr(agent, "json_llm", ScriptedLLM([json.dumps(PLAN)]))
    assert agent.planner({"question": QUESTION})["plan"]["tasks"][0]["tool"] == "both"


def test_full_graph_runs_offline_end_to_end(monkeypatch):
    monkeypatch.setattr(agent, "run_tavily_search", lambda q: [tool_item("alpha"), tool_item("beta")])
    monkeypatch.setattr(agent, "run_arxiv_search",
                        lambda q, max_results=4: [tool_item("gamma", "Academic (arXiv)", "http://arxiv.org/abs/2501.00003v1")])

    class GraphLLM:
        """Planner call first, then a synthesis call that cites whatever evidence IDs appear in the prompt."""
        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
            self.calls += 1

            class Response:
                content = ""
            if self.calls == 1:
                Response.content = json.dumps(PLAN)
            else:
                prompt = messages[-1].content
                ids = sorted({int(x) for x in __import__("re").findall(r"^\[(\d+)\]", prompt, flags=__import__("re").M)})
                Response.content = draft_json([{"claim": SNIPPET.format(w=w), "source_ids": [i]}
                                               for i, w in zip(ids, ("alpha", "beta", "gamma"))])
            return Response()

    monkeypatch.setattr(agent, "json_llm", GraphLLM())
    result = agent.app.invoke({"question": QUESTION, "plan": {}, "evidence": [], "report": {}, "answer": "", "error": None})
    report = result["report"]
    assert len(report["key_findings"]) == 3
    assert report["confidence"] == "Medium"  # 3 distinct verified sources
    assert {s["url"] for s in report["sources"]} <= {e["url"] for e in result["evidence"]}

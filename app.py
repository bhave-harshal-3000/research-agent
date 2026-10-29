import html
import time

import streamlit as st

import runner

st.set_page_config(page_title="AI Research Agent", layout="centered")

CSS = """
html, body, .stApp, [data-testid="stAppViewContainer"], header[data-testid="stHeader"] { background: #fff; color: #000; }
#MainMenu, footer { visibility: hidden; }
.block-container { max-width: 860px; padding-top: 3rem; padding-bottom: 4rem; }
h1 { font-weight: 650; letter-spacing: -0.02em; margin-bottom: 0; }
h2, h3 { font-weight: 600; letter-spacing: -0.01em; }
[data-testid="stCaptionContainer"] { color: #525252; }
textarea { border: 1px solid #d4d4d4 !important; border-radius: 6px !important; background: #fff !important; color: #000 !important; }
textarea:focus { border-color: #000 !important; box-shadow: none !important; }
div.stButton > button { background: #000; color: #fff; border: 1px solid #000; border-radius: 6px; padding: 0.55rem 1.5rem; font-weight: 500; }
div.stButton > button:hover { background: #262626; border-color: #262626; color: #fff; }
div.stButton > button:disabled { background: #e5e5e5; border-color: #e5e5e5; color: #8a8a8a; }
div.stButton > button p { color: inherit; }
.section { border-top: 1px solid #e5e5e5; margin-top: 2rem; padding-top: 0.25rem; }
[data-testid="stCodeBlock"] { border: 1px solid #e5e5e5 !important; border-radius: 6px !important; background: #fafafa !important; }
[data-testid="stCodeBlock"] pre, [data-testid="stCodeBlock"] code {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace !important;
    font-size: 12.5px !important;
    line-height: 1.55 !important;
    color: #111 !important;
    background: transparent !important;
}
.pill { display: inline-block; border: 1px solid #000; border-radius: 999px; padding: 2px 14px; font-weight: 500; }
[data-testid="stMetric"] { border: 1px solid #e5e5e5; border-radius: 6px; padding: 10px 14px; }
[data-testid="stMetricLabel"] { color: #525252; }
[data-testid="stExpander"] { border: 1px solid #e5e5e5; border-radius: 6px; }
a { color: #000; text-decoration: underline; }
"""
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Small rendering helpers
# ---------------------------------------------------------------------------
def md(text: str) -> str:
    """Escape characters that Streamlit's markdown would otherwise interpret (LaTeX '$', link brackets)."""
    return str(text).replace("\\", "\\\\").replace("$", "\\$").replace("[", "\\[").replace("]", "\\]")


def link(title: str, url: str) -> str:
    if url.lower().startswith(("http://", "https://")):
        safe_url = url.replace(" ", "%20").replace("(", "%28").replace(")", "%29")
        return f"[{md(title)}]({safe_url})"
    return md(title)


def log_text(lines: list[str]) -> str:
    return "\n".join(lines) if lines else "Starting... (loading libraries and connecting to services)"


def render_report(report: dict) -> None:
    st.markdown("### Short Answer")
    st.markdown(md(report.get("short_answer", "")))

    st.markdown("### Key Findings")
    findings = report.get("key_findings") or []
    if findings:
        for f in findings:
            ids = ", ".join(str(i) for i in f.get("source_ids", []))
            st.markdown(f"- {md(f.get('claim', ''))} \\[{ids}\\]")
    else:
        st.caption("No verified findings.")

    st.markdown("### Sources")
    sources = report.get("sources") or []
    if sources:
        for s in sources:
            label = f"\\[{s['id']}\\] " if s.get("id") is not None else ""
            st.markdown(f"- {label}{link(s.get('title', 'Source'), s.get('url', ''))}")
    else:
        st.caption("No verified sources.")

    st.markdown("### Confidence")
    st.markdown(f'<span class="pill">{html.escape(str(report.get("confidence", "N/A")))}</span>', unsafe_allow_html=True)

    st.markdown("### Limitations & Assumptions")
    for item in report.get("limitations_and_assumptions") or ["None reported."]:
        st.markdown(f"- {md(item)}")

    st.markdown("### Next Steps")
    for item in report.get("suggested_next_steps") or ["None suggested."]:
        st.markdown(f"- {md(item)}")


def render_summary(run: "runner.ResearchRun") -> None:
    s = runner.summarize(run)
    cols = st.columns(5)
    cols[0].metric("Total time", f"{s['elapsed']:.1f}s")
    cols[1].metric("Tasks completed", f"{s['tasks_done']}/{s['tasks_total']}")
    cols[2].metric("Sources collected", s["sources_collected"])
    cols[3].metric("Sources cited", s["sources_cited"])
    cols[4].metric("Errors / retries", f"{s['errors']} / {s['retries']}")


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
def on_start() -> None:
    question = (st.session_state.get("question") or "").strip()
    st.session_state.input_error = None
    if not question:
        st.session_state.input_error = "Please enter a research question."
        return
    try:
        st.session_state.run = runner.start_run(question)
    except runner.RunBusyError:
        st.session_state.input_error = "A research run is still in progress. Please wait for it to finish."


run = st.session_state.get("run")
running = run is not None and not run.done

st.title("AI Research Agent")
st.caption("Plans the research, searches the web and arXiv, and writes a source-grounded report.")

st.text_area(
    "Research question",
    key="question",
    height=140,
    disabled=running,
    placeholder="e.g. Compare the latest approaches to adding memory to AI agents and explain the tradeoffs.",
)
st.button("Start Research", type="primary", disabled=running, on_click=on_start)

if st.session_state.get("input_error"):
    st.warning(st.session_state.input_error)

if running:
    st.markdown('<div class="section"></div>', unsafe_allow_html=True)
    st.markdown("### Live execution log")
    status = st.empty()
    log_box = st.empty()
    last_count = -1
    while not run.done:
        lines = run.log.snapshot()
        status.caption(f"Running: {runner.current_step(lines)} ({time.time() - run.started:.0f}s elapsed)")
        if len(lines) != last_count:
            log_box.code(log_text(lines), language="text", wrap_lines=True, height=380)
            last_count = len(lines)
        time.sleep(0.4)
    st.rerun()

elif run is not None:
    report = (run.result or {}).get("report") or {}
    st.markdown('<div class="section"></div>', unsafe_allow_html=True)

    if run.error:
        st.error(f"The research run failed. {run.error}")
    elif report:
        st.success("Research complete.")
        render_report(report)

    st.markdown("### Execution summary")
    render_summary(run)

    with st.expander("Execution log", expanded=bool(run.error)):
        st.code(log_text(run.log.snapshot()), language="text", wrap_lines=True, height=380)

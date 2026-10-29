"""
Backend glue for the Streamlit UI.

Runs the existing LangGraph agent (`agent.app`) in a background thread and captures everything it
`print()`s (plus warnings from the logging module) into a thread-safe buffer the UI can poll.
Nothing here re-implements agent logic; it only calls `agent.app.invoke`.
"""
import importlib
import logging
import os
import re
import sys
import threading
import time
import traceback

# Only one run at a time: stdout is process-global, so concurrent runs would mix their logs.
_ACTIVE = threading.Lock()

_SECRET_ENV_VARS = ("GROQ_API_KEY", "TAVILY_API_KEY", "GITHUB_TOKEN")
_SECRET_PATTERNS = (re.compile(r"gsk_[A-Za-z0-9]{10,}"), re.compile(r"tvly-[A-Za-z0-9\-_]{10,}"))


class RunBusyError(RuntimeError):
    """Raised when a research run is already in progress."""


def redact(text: str) -> str:
    """Remove API keys (from the environment or matching known key formats) from text."""
    for name in _SECRET_ENV_VARS:
        value = os.getenv(name)
        if value and len(value) >= 8:
            text = text.replace(value, "[REDACTED]")
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    return text


class RunLog:
    """Thread-safe line buffer fed by print() and logging; the UI reads snapshots."""

    def __init__(self):
        self._lock = threading.Lock()
        self._lines: list[str] = []
        self._partial = ""

    def write(self, text: str) -> None:
        with self._lock:
            *complete, self._partial = (self._partial + text).split("\n")
            self._lines.extend(redact(line) for line in complete)

    def snapshot(self) -> list[str]:
        with self._lock:
            lines = list(self._lines)
            if self._partial:
                lines.append(redact(self._partial))
        return lines


class _StdoutRouter:
    """Temporary sys.stdout replacement that writes into a RunLog; other attributes pass through."""

    def __init__(self, original, sink: RunLog):
        self._original = original
        self._sink = sink

    def write(self, text: str) -> int:
        self._sink.write(text)
        return len(text)

    def flush(self) -> None:
        pass

    def __getattr__(self, name):
        return getattr(self._original, name)


class _LogHandler(logging.Handler):
    """Forwards log records (except Streamlit's own) into the run log."""

    def __init__(self, sink: RunLog):
        super().__init__(level=logging.WARNING)
        self._sink = sink
        self.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        if record.name.startswith(("streamlit", "watchdog", "tornado")):
            return
        try:
            self._sink.write(self.format(record) + "\n")
        except Exception:
            pass


class ResearchRun:
    """State of one research run, shared between the worker thread and the UI."""

    def __init__(self, question: str):
        self.question = question
        self.log = RunLog()
        self.result: dict | None = None
        self.error: str | None = None
        self.started = time.time()
        self.elapsed: float | None = None
        self._finished = threading.Event()

    @property
    def done(self) -> bool:
        return self._finished.is_set()


def friendly_error(exc: Exception) -> str:
    message = redact(str(exc)).strip() or "no further details"
    if len(message) > 600:
        message = message[:600] + "..."
    return f"{type(exc).__name__}: {message}"


def _worker(run: ResearchRun) -> None:
    previous_stdout = sys.stdout
    router = _StdoutRouter(previous_stdout, run.log)
    handler = _LogHandler(run.log)
    root = logging.getLogger()
    sys.stdout = router
    root.addHandler(handler)
    started = time.perf_counter()
    try:
        agent = importlib.import_module("agent")  # imported here so a missing key becomes a readable error
        run.result = agent.app.invoke({
            "question": run.question,
            "plan": {},
            "evidence": [],
            "report": {},
            "answer": "",
            "error": None,
            "coverage_gaps": [],
        })
    except Exception as exc:
        run.error = friendly_error(exc)
        print(f"\n❌ Run failed: {run.error}")
        print(redact(traceback.format_exc()))
    finally:
        run.elapsed = time.perf_counter() - started
        root.removeHandler(handler)
        if sys.stdout is router:
            sys.stdout = previous_stdout
        _ACTIVE.release()
        run._finished.set()


def start_run(question: str) -> ResearchRun:
    """Start the agent in a background thread. Raises RunBusyError if a run is already active."""
    if not _ACTIVE.acquire(blocking=False):
        raise RunBusyError("A research run is already in progress.")
    run = ResearchRun(question)
    try:
        threading.Thread(target=_worker, args=(run,), daemon=True, name="research-run").start()
    except Exception:
        _ACTIVE.release()
        raise
    return run


# ---------------------------------------------------------------------------
# Log parsing helpers (the UI shows what the agent already prints)
# ---------------------------------------------------------------------------
_STEP_RE = re.compile(r"STEP \d:\s*(.+?)\s{2,}\[")
_ERROR_MARKERS = ("[TAVILY ERROR]", "[ARXIV ERROR]", "❌", "Fallback triggered")
_RETRY_MARKERS = ("Retrying", "Shrinking evidence", "Waiting ")


def current_step(lines: list[str]) -> str:
    """Name of the most recent pipeline step found in the log."""
    for line in reversed(lines):
        match = _STEP_RE.search(line)
        if match:
            return match.group(1).title()
    return "Starting"


def summarize(run: ResearchRun) -> dict:
    """Execution summary derived from the final state and the captured log."""
    lines = run.log.snapshot()
    text = "\n".join(lines)
    result = run.result or {}
    report = result.get("report") or {}
    planned = len((result.get("plan") or {}).get("tasks") or [])
    if not planned:
        match = re.search(r"Total Tasks Scheduled: (\d+)", text)
        planned = int(match.group(1)) if match else 0
    raw = re.search(r"Raw Evidence Collected: (\d+) items", text)
    return {
        "elapsed": run.elapsed if run.elapsed is not None else time.time() - run.started,
        "tasks_done": len(re.findall(r"Task \d+ collected \d+ raw evidence items", text)),
        "tasks_total": planned,
        "sources_collected": int(raw.group(1)) if raw else len(result.get("evidence") or []),
        "sources_kept": len(result.get("evidence") or []),
        "sources_cited": len(report.get("sources") or []),
        "errors": sum(1 for line in lines if any(m in line for m in _ERROR_MARKERS)),
        "retries": sum(1 for line in lines if any(m in line for m in _RETRY_MARKERS)),
    }

"""Live progress bus for mid-node events.

LangGraph's stream() only yields when a node RETURNS. That means the whole
search node (~24s for 7 queries) runs silently and dumps one summary line at
the end. Same for extract/fact-check.

This module lets agent code emit progress messages that reach the UI live.
`run_research` sets `_hook` to the caller's on_event callback before invoking
the graph; agents call `emit(msg)` to fire it directly. Since the graph runs
synchronously in the Streamlit script thread, on_event calls st.status.write
which draws immediately.

For parallelized work (search, extract, fact-check), emit from the MAIN
thread as futures complete — not from worker threads, since Streamlit
`add_script_run_ctx` isn't threaded through and cross-thread writes can drop.
"""
from typing import Callable

_hook: Callable[[str], None] | None = None


def set_hook(fn: Callable[[str], None] | None) -> None:
    global _hook
    _hook = fn


def emit(msg: str) -> None:
    if _hook is not None:
        try:
            _hook(msg)
        except Exception:
            pass

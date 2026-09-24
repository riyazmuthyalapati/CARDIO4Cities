"""LangGraph RESEARCH WORKFLOW (write path).

    plan -> search -> crawl_gate -> extract -> fact_check -> judge
                                                               |
                            (insufficient & iterations left)   v
                            plan <---------------------- [conditional]
                                                               |
                                              (sufficient/out of budget)
                                                               v
                                            gap_analysis -> curate -> report

Structural trust guarantees:
- crawl_gate runs BEFORE extract: nothing is fetched without a verdict.
- fact_check sits between extract and everything downstream; curate/report
  read only verified claims. Quarantine is enforced by the graph topology.

Each node emits a progress event via state["events"] so the UI can narrate.
"""
import operator
import threading
from typing import Annotated, TypedDict

from langgraph.graph import END, StateGraph

from src.agents import crawl_gate, extractor, fact_checker, planner, reporter, sufficiency
from src.agents.searcher import run_searches
from src.config import get_settings
from src.models import Claim, CrawlVerdict, Gap, Source
from src.progress import emit, set_hook
from src.stores import relational, vector
from src.stores.graph import add_claim_episode


# ---- Background Graphiti ingest status -------------------------------------
# Graphiti ingest (~15-30s of LLM/embedding calls) runs OFF the critical path
# so the Streamlit UI returns the report immediately. We track per-run status
# here so the graph tab can render "building…" and poll for completion.
_graph_status: dict[int, dict] = {}
_graph_status_lock = threading.Lock()


def _set_graph_status(run_id: int, **fields) -> None:
    with _graph_status_lock:
        cur = _graph_status.setdefault(run_id, {})
        cur.update(fields)


def get_graph_ingest_status(run_id: int) -> dict | None:
    """Return {state, done, total, error} for the background graph ingest of
    `run_id`, or None if we have no record (older run, or process restarted).
    state ∈ {'pending', 'running', 'completed', 'failed'}."""
    with _graph_status_lock:
        cur = _graph_status.get(run_id)
        return dict(cur) if cur is not None else None


def rebuild_graph_background(run_id: int, city: str) -> None:
    """Kick a fresh Graphiti ingest for an existing run. Used when the previous
    ingest was orphaned (Streamlit restarted before the daemon thread finished
    — status dict is in-memory only, so it doesn't survive process death)."""
    _set_graph_status(run_id, state="pending", done=0, total=0, error="")
    t = threading.Thread(
        target=_ingest_graph_background, args=(run_id, city),
        name=f"graphiti-ingest-{run_id}", daemon=True,
    )
    t.start()


class ResearchState(TypedDict):
    city: str
    run_id: int
    iteration: int
    gap_feedback: dict           # judge feedback for re-planning
    queries: list                # current iteration's SearchQuery list
    new_sources: list            # current iteration's Source list
    pending: dict                # extract -> fact_check handoff: {url: [Claim]}
    sources: Annotated[list, operator.add]   # all sources (accumulated)
    claims: Annotated[list, operator.add]    # all claims (accumulated)
    coverage: dict
    gaps: list
    report_md: str
    events: Annotated[list, operator.add]    # progress feed for the UI


def _ev(msg: str) -> dict:
    return {"events": [msg]}


import time as _time


def _timed(label: str):
    """Context manager that stamps a wall-clock line into the events feed."""
    class _T:
        def __enter__(self):
            self.t0 = _time.time()
            return self
        def __exit__(self, *a):
            self.elapsed = _time.time() - self.t0
    return _T()


def plan_node(state: ResearchState) -> dict:
    emit(f"🧭 Planning queries for {state['city']}"
         + (" — targeting gaps" if state.get("gap_feedback") else ""))
    with _timed("plan") as t:
        queries = planner.plan_queries(state["city"], state.get("gap_feedback") or None)
    for q in queries:
        emit(f"🧭   • [{q.dimension}] {q.query}")
    return {"queries": queries,
            **_ev(f"🧭 Planner: {len(queries)} search queries in {t.elapsed:.1f}s")}


def search_node(state: ResearchState) -> dict:
    """Parallelize the queries — Tavily/DDG calls are ~2-3s each, running
    them concurrently drops search from ~24s to ~4s. Emit each query result
    as it lands so the UI narrates live."""
    import concurrent.futures as cf
    from src.agents.searcher import _tavily_search, _ddg_search, _credibility
    from urllib.parse import urlparse
    t0 = _time.time()
    seen = {s.url for s in state.get("sources", [])}
    queries = state["queries"]
    s = get_settings()
    emit(f"🔎 Running {len(queries)} searches in parallel…")

    def _one(q):
        try:
            r = _tavily_search(q.query, s.max_results_per_query)
        except Exception:
            try:
                r = _ddg_search(q.query, s.max_results_per_query)
            except Exception:
                r = []
        return q, r

    found: list[Source] = []
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(_one, q): q for q in queries}
        for fut in cf.as_completed(futures):
            q, results = fut.result()
            n_new = 0
            for r in results:
                url = r["url"].split("#")[0]
                if url in seen:
                    continue
                seen.add(url)
                domain = urlparse(url).netloc
                found.append(Source(
                    url=url, domain=domain, title=r["title"], snippet=r["snippet"],
                    dimension=q.dimension, credibility_tier=_credibility(domain),
                ))
                n_new += 1
            emit(f"🔎   ✓ [{q.dimension}] {n_new} results")
    return {"new_sources": found,
            **_ev(f"🔎 Searcher: {len(found)} new candidate sources in {_time.time()-t0:.1f}s")}


CRED_RANK = {"high": 0, "medium": 1, "unrated": 2}


def crawl_gate_node(state: ResearchState) -> dict:
    """Verdict every source in parallel (robots.txt fetches were the bulk of
    the sequential wait), then keep the top-N per dimension (allowed +
    higher-credibility first) to bound downstream LLM calls."""
    import concurrent.futures as cf
    t0 = _time.time()
    cap = get_settings().max_sources_per_dimension
    gated: list[Source] = []
    events = []
    sources = list(state["new_sources"])
    emit(f"🛂 Checking robots.txt + pre-fetching {len(sources)} pages in parallel…")
    t_fetch = _time.time()
    results = [None] * len(sources)
    done = 0
    with cf.ThreadPoolExecutor(max_workers=10) as ex:
        futures = {ex.submit(crawl_gate.check_crawlability, s): idx
                   for idx, s in enumerate(sources)}
        for fut in cf.as_completed(futures):
            idx = futures[fut]
            src = fut.result()
            results[idx] = src
            done += 1
            if src.raw_text:
                body = " · body ok"
            elif "body-fetch:" in src.robots_evidence:
                # Surface the reason so "allowed but 0 claims" is explicable
                reason = src.robots_evidence.rsplit("body-fetch:", 1)[-1].strip()
                body = f" · no body ({reason})"
            else:
                body = ""
            emit(f"🛂   [{done}/{len(sources)}] {src.domain}: {src.crawl_verdict.value}{body}")
    fetch_s = _time.time() - t_fetch
    t_db = _time.time()
    # Bulk insert all sources in one transaction — one round-trip, not N.
    source_ids = relational.save_sources(state["run_id"], results)
    for src, sid in zip(results, source_ids):
        src.db_id = sid
        gated.append(src)
    db_s = _time.time() - t_db

    # Prune: allowed sources first, then higher-credibility, then discovery order
    def _rank(s: Source) -> tuple:
        return (0 if s.crawl_verdict == CrawlVerdict.ALLOWED else 1,
                CRED_RANK.get(s.credibility_tier, 3))
    kept: list[Source] = []
    by_dim: dict[str, list[Source]] = {}
    for s in gated:
        by_dim.setdefault(s.dimension, []).append(s)
    for dim, items in by_dim.items():
        items.sort(key=_rank)
        kept.extend(items[:cap])

    n_ok = sum(1 for s in kept if s.crawl_verdict == CrawlVerdict.ALLOWED)
    n_bodies = sum(1 for s in kept if s.raw_text)
    events.append(
        f"🛂 Crawl gate: {len(gated)} checked → top {len(kept)} kept "
        f"({n_ok} allowed, {len(kept)-n_ok} snippet-only, {n_bodies} bodies pre-fetched) "
        f"[robots+fetch {fetch_s:.1f}s + DB {db_s:.1f}s]"
    )
    return {"new_sources": kept, "sources": kept, "events": events}


def extract_node(state: ResearchState) -> dict:
    """Extract claims per source in parallel — page bodies were pre-fetched
    by crawl_gate, so this node is pure LLM. Emit as each future completes
    so the UI narrates live instead of dumping after all workers finish."""
    import concurrent.futures as cf
    t0 = _time.time()
    per_source: dict[str, list[Claim]] = {}
    sources = list(state["new_sources"])
    total = len(sources)
    emit(f"📄 Extracting claims from {total} sources in parallel…")

    def _one(src):
        return src, extractor.extract_claims(state["city"], src)

    done = 0
    total_claims = 0
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(_one, s): idx for idx, s in enumerate(sources, 1)}
        for fut in cf.as_completed(futures):
            done += 1
            try:
                src, cs = fut.result()
                per_source[src.url] = cs
                total_claims += len(cs)
                emit(f"📄   [{done}/{total}] {src.domain}: {len(cs)} claim(s)")
            except Exception:
                emit(f"📄   [{done}/{total}] <error>")
    return {"pending": per_source,
            **_ev(f"📄 Extract: {total_claims} claims from {total} sources "
                  f"[{_time.time()-t0:.1f}s]")}


def fact_check_node(state: ResearchState) -> dict:
    """Fact-check per source in parallel — emits live as each source's
    check completes, and inserts to Postgres on the main thread as futures
    land (single-connection safety, but no longer batched to the end)."""
    import concurrent.futures as cf
    t0 = _time.time()
    checked: list[Claim] = []
    src_by_url = {s.url: s for s in state["new_sources"]}
    items = list(state.get("pending", {}).items())
    total = len(items)
    emit(f"🕵️ Fact-checking claims from {total} sources in parallel…")

    def _one(url, claims):
        src = src_by_url[url]
        verified = fact_checker.check_claims(state["city"], src, claims)
        return src, verified

    # Buffer all verified claims and their source_ids across all futures; one
    # bulk insert at the end. Downstream nodes (judge, gap_analysis, curate)
    # only read c.db_id inside curate — safe to defer assignment until then.
    pending_rows: list[tuple[Claim, int | None]] = []
    done = 0
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(_one, url, claims): (url, claims)
                   for url, claims in items}
        for fut in cf.as_completed(futures):
            done += 1
            try:
                src, verified = fut.result()
            except Exception:
                emit(f"🕵️   [{done}/{total}] <error>")
                continue
            pending_rows.extend((c, src.db_id) for c in verified)
            q = sum(1 for c in verified if c.quarantined)
            emit(f"🕵️   [{done}/{total}] {src.domain}: "
                 f"{len(verified)-q}/{len(verified)} verified"
                 + (f", {q} quarantined" if q else ""))
            checked.extend(verified)

    # One bulk insert for every claim in this iteration — one round-trip, one commit.
    claim_ids = relational.save_claims(state["run_id"], pending_rows)
    for (c, _), cid in zip(pending_rows, claim_ids):
        c.db_id = cid

    q_total = sum(1 for c in checked if c.quarantined)
    nat = sum(1 for c in checked if c.verdict and c.verdict.value == "national_not_city")
    return {"claims": checked,
            **_ev(f"🕵️ Fact checker: {len(checked)} audited — "
                  f"{len(checked) - q_total} verified, {q_total} quarantined, "
                  f"{nat} flagged national-level [{_time.time()-t0:.1f}s]")}


def judge_node(state: ResearchState) -> dict:
    scores, missing = sufficiency.judge_sufficiency(state["city"], state["claims"])
    low = {k: v for k, v in missing.items()
           if scores.get(k, 0) < get_settings().coverage_threshold}
    avg = sum(scores.values()) / max(len(scores), 1)
    return {"coverage": scores, "gap_feedback": low,
            "iteration": state["iteration"] + 1,
            **_ev(f"⚖️ Sufficiency judge: avg coverage {avg:.0%}; "
                  f"{len(low)} dimension(s) below threshold")}


def should_iterate(state: ResearchState) -> str:
    if state["gap_feedback"] and state["iteration"] < get_settings().max_iterations:
        return "plan"
    return "gap_analysis"


def gap_analysis_node(state: ResearchState) -> dict:
    gaps: list[Gap] = sufficiency.analyse_gaps(
        state["city"], state["coverage"], state["gap_feedback"], state["claims"])
    relational.save_gaps(state["run_id"], gaps)
    return {"gaps": gaps, **_ev(f"🕳️ Gap analyst: {len(gaps)} known unknowns recorded")}


GRAPH_CAP = 8  # top-N claims sent to Graphiti; the rest live only in Qdrant


def curate_node(state: ResearchState) -> dict:
    """Knowledge Curator — Qdrant (vector) upsert only. Graphiti ingest is
    a separate node that runs AFTER report generation so users don't wait on
    a ~60-100s LLM/embedding pass to see their briefing.
    """
    t_curate = _time.time()
    verified = [c for c in state["claims"] if c.is_verified]

    vector.upsert_evidence(state["run_id"], [
        {"text": f"{c.statement}\nEvidence: {c.exact_quote}",
         "claim_id": c.db_id, "source_url": c.source_url,
         "dimension": c.dimension, "scope": c.scope.value,
         "verdict": c.verdict.value}
        for c in verified
    ])
    return {**_ev(
        f"🧠 Curator: {len(verified)} verified claims → vector store "
        f"[{_time.time()-t_curate:.1f}s]"
    )}


SCOPE_PRIORITY = {"city": 0, "regional": 1, "national": 2, "unknown": 3}


def _rank_claims_for_graph(claims: list) -> list:
    """Round-robin by dimension so we don't ingest 6 rewordings of the same
    fact and collapse the graph to 2 entities. Within a dimension, city-scope
    facts come first. Accepts either Claim objects (from in-memory state) or
    dict rows (from relational.get_claims)."""
    def _dim(c):
        return c.dimension if hasattr(c, "dimension") else c["dimension"]

    def _scope(c):
        v = c.scope.value if hasattr(c, "scope") else c["scope"]
        return SCOPE_PRIORITY.get(v, 4)

    by_dim: dict[str, list] = {}
    for c in claims:
        by_dim.setdefault(_dim(c), []).append(c)
    for lst in by_dim.values():
        lst.sort(key=_scope)
    ranked: list = []
    while any(by_dim.values()):
        for dim in list(by_dim.keys()):
            if by_dim[dim]:
                ranked.append(by_dim[dim].pop(0))
    return ranked


def _ingest_graph_background(run_id: int, city: str) -> None:
    """Runs on a daemon thread AFTER run_research returns. Reads verified
    claims from Postgres (not from in-memory state — decoupled from workflow
    lifecycle), ingests up to GRAPH_CAP into Graphiti with 4-way concurrency,
    and records progress via _set_graph_status so the UI graph tab can poll."""
    import concurrent.futures as cf

    try:
        rows = relational.get_claims(run_id, verified_only=True)
    except Exception as e:
        _set_graph_status(run_id, state="failed", done=0, total=0,
                          error=f"{type(e).__name__}: {str(e)[:200]}")
        return

    ranked = _rank_claims_for_graph(rows)
    cap = min(GRAPH_CAP, len(ranked))
    _set_graph_status(run_id, state="running", done=0, total=cap, error="")

    if cap == 0:
        _set_graph_status(run_id, state="completed", done=0, total=0)
        return

    def _ingest(row: dict) -> tuple[bool, str]:
        try:
            add_claim_episode(run_id, city, row["statement"],
                              row["source_url"], row["dimension"])
            return True, ""
        except Exception as e:
            return False, f"{type(e).__name__}: {str(e)[:200]}"

    ok_count = 0
    first_err = ""
    done = 0
    with cf.ThreadPoolExecutor(max_workers=4) as ex:
        futures = [ex.submit(_ingest, r) for r in ranked[:cap]]
        for fut in cf.as_completed(futures):
            ok, err = fut.result()
            done += 1
            if ok:
                ok_count += 1
            elif not first_err:
                first_err = err
            _set_graph_status(run_id, done=done)

    _set_graph_status(
        run_id,
        state="completed" if ok_count > 0 or not first_err else "failed",
        done=done, total=cap, error=first_err if ok_count == 0 else "",
    )


def report_node(state: ResearchState) -> dict:
    claims = relational.get_claims(state["run_id"])
    gaps = relational.get_gaps(state["run_id"])
    md = reporter.generate_report(state["city"], claims, gaps)
    relational.finish_run(state["run_id"], "completed", state["iteration"],
                          state["coverage"], md)
    return {"report_md": md, **_ev("📋 Report generated — research complete")}


def build_research_graph():
    g = StateGraph(ResearchState)
    g.add_node("plan", plan_node)
    g.add_node("search", search_node)
    g.add_node("crawl_gate", crawl_gate_node)
    g.add_node("extract", extract_node)
    g.add_node("fact_check", fact_check_node)
    g.add_node("judge", judge_node)
    g.add_node("gap_analysis", gap_analysis_node)
    g.add_node("curate", curate_node)
    g.add_node("report", report_node)

    g.set_entry_point("plan")
    g.add_edge("plan", "search")
    g.add_edge("search", "crawl_gate")
    g.add_edge("crawl_gate", "extract")
    g.add_edge("extract", "fact_check")
    g.add_edge("fact_check", "judge")
    g.add_conditional_edges("judge", should_iterate,
                            {"plan": "plan", "gap_analysis": "gap_analysis"})
    g.add_edge("gap_analysis", "curate")
    g.add_edge("curate", "report")
    # Graphiti ingest is dispatched off the DAG by run_research so the report
    # returns to the UI immediately; graph tab fills in over the next ~15-30s.
    g.add_edge("report", END)
    return g.compile()


def run_research(city: str, on_event=None):
    """Execute the workflow, streaming progress events to on_event(msg).

    Two channels for events:
    1. Live intra-node updates via src.progress.emit() (fires immediately).
    2. Per-node summary events via state['events'] (fires when node returns).
    Both go through the same on_event so the UI sees a continuous feed.
    """
    run_id = relational.create_run(city)
    # De-dupe: intra-node emit() already fired the summary line via events[]
    # would double-print. Track what emit() has shown so the post-return
    # summary flush doesn't repeat it.
    seen: set[str] = set()

    def _forward(msg: str) -> None:
        if on_event and msg not in seen:
            seen.add(msg)
            on_event(msg)

    set_hook(_forward)
    try:
        graph = build_research_graph()
        state = {"city": city, "run_id": run_id, "iteration": 0, "gap_feedback": {},
                 "queries": [], "new_sources": [], "pending": {}, "sources": [],
                 "claims": [], "coverage": {}, "gaps": [], "report_md": "", "events": []}
        for chunk in graph.stream(state, {"recursion_limit": 60}):
            for node_state in chunk.values():
                for ev in (node_state or {}).get("events", []):
                    _forward(ev)
    finally:
        set_hook(None)

    # Fire Graphiti ingest on a daemon thread — the biggest tail cost
    # (~15-30s of LLM + embedding calls) now runs AFTER we return, so the
    # user sees their briefing immediately. The graph tab polls
    # get_graph_ingest_status(run_id) and shows a "building…" state.
    _set_graph_status(run_id, state="pending", done=0, total=0, error="")
    _forward("🕸️ Graphiti ingest dispatched — graph tab will populate in the background")
    t = threading.Thread(
        target=_ingest_graph_background, args=(run_id, city),
        name=f"graphiti-ingest-{run_id}", daemon=True,
    )
    t.start()
    return run_id

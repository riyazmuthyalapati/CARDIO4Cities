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
from typing import Annotated, TypedDict

from langgraph.graph import END, StateGraph

from src.agents import crawl_gate, extractor, fact_checker, planner, reporter, sufficiency
from src.agents.searcher import run_searches
from src.config import get_settings
from src.models import Claim, CrawlVerdict, Gap, Source
from src.progress import emit, set_hook
from src.stores import relational, vector
from src.stores.graph import add_claim_episode


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
            body = " · body ok" if src.raw_text else ""
            emit(f"🛂   [{done}/{len(sources)}] {src.domain}: {src.crawl_verdict.value}{body}")
    fetch_s = _time.time() - t_fetch
    t_db = _time.time()
    for src in results:
        # DB insert stays single-threaded to keep the psycopg connection safe
        src.db_id = relational.save_source(state["run_id"], src)
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
            # Insert on main thread — psycopg connection isn't thread-safe
            for c in verified:
                c.db_id = relational.save_claim(state["run_id"], c, src.db_id)
            q = sum(1 for c in verified if c.quarantined)
            emit(f"🕵️   [{done}/{total}] {src.domain}: "
                 f"{len(verified)-q}/{len(verified)} verified"
                 + (f", {q} quarantined" if q else ""))
            checked.extend(verified)

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


def curate_node(state: ResearchState) -> dict:
    """Knowledge Curator — receives ONLY verified claims (graph topology
    guarantees quarantined content never reaches the knowledge asset).

    Graphiti's entity extraction consumes 2-3 Gemini calls per episode, so we
    cap the graph ingest to the top N claims per run (prioritising higher-scope
    facts) to stay under free-tier RPM. Any Graphiti exception is logged, not
    swallowed, so we don't silently end up with 0 episodes again.
    """
    t_curate = _time.time()
    verified = [c for c in state["claims"] if c.is_verified]

    # 1. Vector store — cheap, always full set
    vector.upsert_evidence(state["run_id"], [
        {"text": f"{c.statement}\nEvidence: {c.exact_quote}",
         "claim_id": c.db_id, "source_url": c.source_url,
         "dimension": c.dimension, "scope": c.scope.value,
         "verdict": c.verdict.value}
        for c in verified
    ])
    events: list[str] = [f"🧠 Curator: {len(verified)} verified claims → vector store"]

    # 2. Graphiti — round-robin by dimension so we don't ingest 6 rewordings
    # of the same fact and collapse the graph to 2 entities. Within a
    # dimension, city-scope facts come first.
    SCOPE_PRIORITY = {"city": 0, "regional": 1, "national": 2, "unknown": 3}
    by_dim: dict[str, list[Claim]] = {}
    for c in verified:
        by_dim.setdefault(c.dimension, []).append(c)
    for lst in by_dim.values():
        lst.sort(key=lambda c: SCOPE_PRIORITY.get(c.scope.value, 4))
    ranked: list[Claim] = []
    while any(by_dim.values()):
        for dim in list(by_dim.keys()):
            if by_dim[dim]:
                ranked.append(by_dim[dim].pop(0))
    # Cap at 20 verified claims. Each episode is ~5-8s (LLM entity extraction +
    # embed + Neo4j write) so this pushes ingest to ~60-100s wall — the graph
    # is mandatory and load-bearing at query time, so we spend the wall time.
    cap = min(20, len(ranked))
    # Concurrent ingest: we tested 4 parallel episodes on the shared persistent
    # asyncio loop — no races, no duplicate entities, ~3x wall speedup
    # (32s vs 92s for 4 episodes). Keep concurrency modest so Graphiti's
    # entity-dedup step isn't fighting itself and aicredits isn't rate-limited.
    import concurrent.futures as cf
    graph_ok = 0
    first_err: str = ""
    emit(f"🕸️ Ingesting {cap} episodes into Graphiti (4 concurrent, ~30-100s)…")

    def _ingest(c: Claim) -> tuple[bool, str, str]:
        try:
            add_claim_episode(state["run_id"], state["city"], c.statement,
                              c.source_url, c.dimension)
            return True, "", c.statement
        except Exception as e:
            return False, f"{type(e).__name__}: {str(e)[:120]}", c.statement

    done = 0
    with cf.ThreadPoolExecutor(max_workers=4) as ex:
        futures = [ex.submit(_ingest, c) for c in ranked[:cap]]
        for fut in cf.as_completed(futures):
            done += 1
            ok, err, stmt = fut.result()
            if ok:
                graph_ok += 1
                emit(f"🕸️   [{done}/{cap}] ✓ {stmt[:60]}")
            else:
                if not first_err:
                    first_err = err
                emit(f"🕸️   [{done}/{cap}] ✗ {err}")

    events.append(
        f"🕸️ Graphiti: {graph_ok}/{cap} episodes ingested into knowledge graph "
        f"[{_time.time()-t_curate:.1f}s wall, 4 concurrent]"
        + (f" (skipped {len(verified) - cap} lower-priority claims)" if len(verified) > cap else "")
    )
    if first_err and graph_ok == 0:
        events.append(f"⚠️ Graphiti ingest error (first): {first_err}")
    return {"events": events}


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
    return run_id

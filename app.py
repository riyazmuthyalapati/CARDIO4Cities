"""CARDIO4Cities — City Intelligence Engine (Streamlit entrypoint)."""
import streamlit as st

st.set_page_config(page_title="CARDIO4Cities · City Intelligence",
                   page_icon="🫀", layout="wide")

from src.config import get_settings, load_streamlit_secrets  # noqa: E402

load_streamlit_secrets()

from src.graph.query_workflow import answer_question  # noqa: E402
from src.graph.research_workflow import run_research  # noqa: E402
from src.stores import relational  # noqa: E402


@st.cache_resource(show_spinner=False)
def bootstrap() -> bool:
    """One-time init of all three datastores."""
    relational.init_schema()
    # Any run stuck in 'running' for >30 min was orphaned — flip to 'failed'
    # so the sidebar dropdown stops showing zombies.
    try:
        relational.mark_stale_runs_failed(older_than_minutes=30)
    except Exception:
        pass
    from src.stores import vector
    vector.init_collection()
    try:
        from src.stores.graph import init_graph
        init_graph()
    except Exception as e:
        st.warning(f"Knowledge graph init deferred: {e}")
    return True


bootstrap()

st.title("🫀 CARDIO4Cities — City Intelligence Engine")
st.caption("Live AI research on any city's cardiovascular health landscape — "
           "every fact verified, every source traceable, every gap stated.")

# ---------- Sidebar: start research / pick a past run ----------
with st.sidebar:
    st.header("Research a city")
    city = st.text_input("City name", placeholder="e.g. Ho Chi Minh City")
    start = st.button("🚀 Start live research", type="primary",
                      use_container_width=True, disabled=not city)

    st.divider()
    st.header("Past research")
    runs = relational.list_runs()
    STATUS_ICON = {"completed": "✅", "running": "⏳", "failed": "❌"}

    def _fmt_dur(r: dict) -> str:
        if not r.get("finished_at"):
            return "…"
        secs = (r["finished_at"] - r["started_at"]).total_seconds()
        if secs < 60:
            return f"{secs:.0f}s"
        return f"{secs/60:.1f}m"

    options: dict[str, tuple[int, str]] = {}
    for r in runs:
        icon = STATUS_ICON.get(r["status"], "•")
        label = f"{icon} #{r['id']} {r['city']} · {_fmt_dur(r)}"
        options[label] = (r["id"], r["city"])
    picked = st.selectbox("Load a past run", ["—"] + list(options))
    if picked != "—":
        st.session_state["run_id"], st.session_state["city"] = options[picked]

if start and city:
    st.subheader(f"Researching {city} — live")
    st.info("The agent workflow narrates every step below: planning, searching, "
            "crawlability checks, extraction, independent fact-checking, and curation.")

    # Group live events into per-node collapsible status widgets. Each event
    # starts with an emoji that maps to the node it came from — we open a new
    # status when a new emoji shows up and close the prior one.
    NODE_ORDER = [
        ("🧭", "plan", "Planner"),
        ("🔎", "search", "Searcher"),
        ("🛂", "crawl_gate", "Crawlability gate"),
        ("📄", "extract", "Extractor"),
        ("🕵️", "fact_check", "Independent fact-checker"),
        ("⚖️", "judge", "Sufficiency judge"),
        ("🕳️", "gap_analysis", "Gap analyst"),
        ("🧠", "curate_vec", "Curator · vector"),
        ("🕸️", "curate_graph", "Curator · knowledge graph"),
        ("⚠️", "warning", "Warning"),
        ("📋", "report", "Report"),
    ]
    EMOJI_TO_LABEL = {emoji: label for emoji, _, label in NODE_ORDER}

    def _emoji(msg: str) -> str:
        for e in EMOJI_TO_LABEL:
            if msg.startswith(e):
                return e
        return ""

    feed = st.container()
    state: dict = {"current_emoji": None, "current_status": None,
                   "opened": {}, "count": 0}

    def on_event(msg: str) -> None:
        emoji = _emoji(msg)
        # Same node as before → append to the open status
        if emoji and emoji == state["current_emoji"] and state["current_status"] is not None:
            state["current_status"].write(msg)
            return
        # Node change → close previous, open new
        if state["current_status"] is not None:
            # keep prior nodes visibly complete
            state["current_status"].update(state="complete")
        label = EMOJI_TO_LABEL.get(emoji, "Step")
        with feed:
            status = st.status(f"{emoji}  {label}", expanded=True)
        status.write(msg)
        state["current_emoji"] = emoji
        state["current_status"] = status
        state["count"] += 1

    try:
        run_id = run_research(city, on_event=on_event)
        # Close the last status widget cleanly
        if state["current_status"] is not None:
            state["current_status"].update(state="complete")
        st.session_state["run_id"] = run_id
        st.session_state["city"] = city
        st.session_state.pop("chat", None)
        st.success("Research complete — explore the tabs below.")
    except Exception as e:
        if state["current_status"] is not None:
            state["current_status"].update(state="error")
        st.error(f"Research failed: {e}")

# ---------- Main: results for the selected run ----------
run_id = st.session_state.get("run_id")
if not run_id:
    st.info("⬅️ Enter a city and start live research, or load a past run.")
    st.stop()

run = relational.get_run(run_id)
city = st.session_state.get("city") or (run["city"] if run else "")
claims = relational.get_claims(run_id)
verified = [c for c in claims if not c["quarantined"] and c["verdict"] != "unsupported"]
gaps = relational.get_gaps(run_id)

tab_dash, tab_ask, tab_graph, tab_evidence, tab_report = st.tabs(
    ["📊 Dashboard", "💬 Ask", "🕸️ Explore graph", "🔍 Evidence ledger", "📋 Report"])

with tab_dash:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Verified facts", len(verified))
    c2.metric("Quarantined claims", sum(1 for c in claims if c["quarantined"]),
              help="Extracted but failed independent fact-checking — never shown as fact")
    c3.metric("National-level flags",
              sum(1 for c in verified if c["scope"] == "national"),
              help="Facts backed by national (not city) data — always flagged")
    c4.metric("Known gaps", len(gaps))

    coverage = (run or {}).get("coverage_scores") or {}
    if coverage:
        st.subheader("Research coverage by dimension")
        cols = st.columns(len(coverage))
        for col, (dim, score) in zip(cols, coverage.items()):
            col.progress(min(float(score), 1.0), text=dim.replace("_", " "))

    st.subheader("What our research pass didn't surface")
    st.caption("Gaps in OUR retrieval — not claims about the world. "
               "These are first-class outputs: stated gaps beat guessed answers.")
    if gaps:
        for g in gaps:
            icon = {"high": "🔴", "medium": "🟠", "low": "🟡"}.get(g["severity"], "🟠")
            st.markdown(f"{icon} **{g['dimension'].replace('_', ' ')}** — {g['description']}")
    else:
        st.write("No gaps recorded.")

    st.subheader("Verified facts by dimension")
    dims = sorted({c["dimension"] for c in verified})
    for dim in dims:
        with st.expander(f"{dim.replace('_', ' ').title()} "
                         f"({sum(1 for c in verified if c['dimension'] == dim)})"):
            for c in [c for c in verified if c["dimension"] == dim]:
                flag = " ⚠️ *national data, not city-specific*" if c["scope"] == "national" else ""
                st.markdown(f"- {c['statement']}{flag}  \n"
                            f"  <sub>📎 [{c['source_url']}]({c['source_url']}) · "
                            f"fact-check: {c['verdict']}</sub>", unsafe_allow_html=True)

with tab_ask:
    st.caption("Answers come only from verified research evidence, with citations. "
               "If we don't know, we say so.")
    if "chat" not in st.session_state:
        st.session_state["chat"] = []
    for turn in st.session_state["chat"]:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn.get("category"):
                st.caption(f"🧭 Router: `{turn['category']}` · "
                           f"{len(turn.get('citations', []))} claim citations · "
                           f"{len(turn.get('graph_facts', []))} graph facts")
            for i, e in enumerate(turn.get("citations", []), 1):
                with st.expander(f"[{i}] 📎 {e['statement'][:80]}…"):
                    st.markdown(f"> {e['exact_quote']}")
                    st.markdown(f"**Source:** [{e['source_url']}]({e['source_url']})  \n"
                                f"**Crawl verdict:** {e['crawl_verdict']} · "
                                f"**Fact-check:** {e['verdict']} — {e['checker_rationale']}  \n"
                                f"**Scope:** {e['scope']}")
            for i, f in enumerate(turn.get("graph_facts", []), 1):
                with st.expander(f"[G{i}] 🕸️ {f['fact'][:80]}…"):
                    st.markdown(f"> {f['fact']}")
                    valid = f.get("valid_at")
                    if valid:
                        st.markdown(f"**Valid at:** {valid}")
                    st.caption("From the Graphiti knowledge graph — extracted "
                               "relationship over verified claim episodes.")

    if question := st.chat_input(f"Ask about {city}…"):
        st.session_state["chat"].append({"role": "user", "content": question})
        with st.spinner("Retrieving evidence…"):
            try:
                result = answer_question(run_id, city, question)
                st.session_state["chat"].append({
                    "role": "assistant", "content": result["answer"],
                    "citations": result["cited_claims"],
                    "graph_facts": result.get("graph_facts", []),
                    "category": result.get("category", ""),
                })
            except Exception as e:
                st.session_state["chat"].append(
                    {"role": "assistant", "content": f"Query failed: {e}"})
        st.rerun()

with tab_graph:
    st.caption("Knowledge graph built by Graphiti: entities and relationships "
               "extracted from verified claims. Click and drag to explore.")

    # Graphiti ingest runs on a background thread after the report — surface
    # its progress so the tab is honest about "still building" vs "no data".
    from src.graph.research_workflow import get_graph_ingest_status
    ingest = get_graph_ingest_status(run_id)
    if ingest and ingest.get("state") in ("pending", "running"):
        done, total = ingest.get("done", 0), ingest.get("total", 0)
        st.info(f"🕸️ Building the knowledge graph in the background — "
                f"{done}/{total or '?'} episodes ingested. "
                f"The rest of the app is already usable.")
        st.progress((done / total) if total else 0.0)
        if st.button("Refresh graph"):
            st.rerun()
    elif ingest and ingest.get("state") == "failed" and ingest.get("error"):
        st.warning(f"Graphiti ingest failed: {ingest['error']}")

    try:
        from src.stores.graph import get_graph_snapshot
        nodes, edges = get_graph_snapshot(run_id)
        if not nodes:
            if not ingest or ingest.get("state") == "completed":
                st.info("No graph entities for this run yet.")
        else:
            from streamlit_agraph import Config, Edge, Node, agraph
            agraph(
                nodes=[Node(id=n["id"], label=n["label"], size=18) for n in nodes],
                edges=[Edge(source=e["source"], target=e["target"], label=e["label"])
                       for e in edges],
                config=Config(width=1100, height=600, directed=True, physics=True),
            )
    except Exception as e:
        st.warning(f"Graph view unavailable: {e}")

with tab_evidence:
    st.caption("The full audit trail — including quarantined claims, kept for "
               "transparency but never presented as facts.")
    show_quarantined = st.toggle("Show quarantined claims", value=False)
    rows = claims if show_quarantined else verified
    st.dataframe(
        [{"statement": c["statement"], "verdict": c["verdict"],
          "scope": c["scope"], "quarantined": c["quarantined"],
          "source": c["source_url"], "crawl": c["crawl_verdict"],
          "quote": (c["exact_quote"] or "")[:120]} for c in rows],
        use_container_width=True, height=500,
    )

with tab_report:
    report_md = (run or {}).get("report_md") or ""
    if report_md:
        st.download_button("⬇️ Download report (Markdown)", report_md,
                           file_name=f"{city.replace(' ', '_')}_briefing.md",
                           mime="text/markdown", type="primary")
        st.markdown(report_md)
    else:
        st.info("No report for this run yet.")

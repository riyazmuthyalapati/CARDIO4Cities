"""Knowledge graph (Graphiti over Neo4j AuraDB) — relationships and time.

Verified claims are ingested as Graphiti *episodes*; Graphiti extracts entities
(people, organisations, programmes, policies, indicators) and temporal
relationships. This layer answers multi-hop questions ("who runs the programme
funded by policy X?") and doubles as institutional memory: re-researching a
city adds facts with new validity intervals instead of overwriting.

Uses Gemini for Graphiti's extraction + embeddings (free tier).
Each research run is isolated via group_id = "run-{run_id}".
"""
import asyncio
import threading
from datetime import datetime, timezone

from src.config import get_settings

_graphiti = None
_loop: asyncio.AbstractEventLoop | None = None
_loop_thread: threading.Thread | None = None
_loop_lock = threading.Lock()


def _ensure_loop() -> asyncio.AbstractEventLoop:
    """Start (once) a dedicated background thread running an asyncio loop.

    Graphiti wraps an async Neo4j driver whose internal pool binds to the
    loop that first touched it — subsequent calls on a different loop error
    with 'Future attached to a different loop'. So we run every Graphiti
    call on the SAME persistent loop for the process lifetime, regardless of
    which thread invoked _run() (Streamlit worker, LangGraph node, CLI).
    """
    global _loop, _loop_thread
    with _loop_lock:
        if _loop is not None and _loop.is_running():
            return _loop
        _loop = asyncio.new_event_loop()

        def _runner(loop: asyncio.AbstractEventLoop) -> None:
            asyncio.set_event_loop(loop)
            loop.run_forever()

        _loop_thread = threading.Thread(target=_runner, args=(_loop,),
                                        name="graphiti-loop", daemon=True)
        _loop_thread.start()
        return _loop


def _get_graphiti():
    global _graphiti
    if _graphiti is not None:
        return _graphiti

    from graphiti_core import Graphiti
    from graphiti_core.cross_encoder.gemini_reranker_client import GeminiRerankerClient
    from graphiti_core.driver.neo4j_driver import Neo4jDriver
    from graphiti_core.embedder.gemini import GeminiEmbedder, GeminiEmbedderConfig
    from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
    from graphiti_core.llm_client.config import LLMConfig
    from graphiti_core.llm_client.gemini_client import GeminiClient
    from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient

    s = get_settings()

    # Neo4jDriver instantiates an AsyncGraphDatabase driver in __init__, which
    # captures the current running loop. Build it on the dedicated loop so it
    # binds to the same loop every call uses.
    loop = _ensure_loop()

    # Prefer aicredits (paid, OpenAI-compatible) for Graphiti's LLM + embedding
    # paths when configured — the free tier's Gemini 15 RPM and Groq 30 RPM
    # both bottleneck here (~3 LLM + ~20 embedding calls per episode →
    # cascading 429s). Aicredits sidesteps the rate limit entirely.
    use_aicredits_llm = bool(s.aicredits_api_key and s.aicredits_model)
    use_aicredits_emb = bool(s.aicredits_api_key and s.aicredits_embedding_model)

    def _build():
        if use_aicredits_llm:
            llm_config = LLMConfig(
                api_key=s.aicredits_api_key,
                model=s.aicredits_model,
                small_model=s.aicredits_model,
                base_url=s.aicredits_base_url,
            )
            llm_client = OpenAIGenericClient(config=llm_config)
        else:
            # Fallback: Gemini. small_model must be pinned or Graphiti's
            # extract_nodes route falls back to stale gemini-2.5-flash-lite
            # (404 for post-deprecation accounts).
            llm_config = LLMConfig(api_key=s.google_api_key, model=s.gemini_model,
                                   small_model=s.gemini_model)
            llm_client = GeminiClient(config=llm_config)

        if use_aicredits_emb:
            embedder = OpenAIEmbedder(config=OpenAIEmbedderConfig(
                api_key=s.aicredits_api_key,
                base_url=s.aicredits_base_url,
                embedding_model=s.aicredits_embedding_model,
                embedding_dim=s.aicredits_embedding_dim,
            ))
        else:
            embedder = GeminiEmbedder(config=GeminiEmbedderConfig(
                api_key=s.google_api_key,
                embedding_model=s.embedding_model,
            ))

        # Cross-encoder always stays on Gemini (avoids OpenAI default that
        # needs OPENAI_API_KEY and is unrelated to the LLM swap).
        gemini_config = LLMConfig(api_key=s.google_api_key, model=s.gemini_model,
                                  small_model=s.gemini_model)

        # AuraDB Free names its default DB after the instance ID, not "neo4j" — inject via driver
        driver = Neo4jDriver(
            uri=s.neo4j_uri, user=s.neo4j_user, password=s.neo4j_password,
            database=s.neo4j_database or "neo4j",
        )
        return Graphiti(
            graph_driver=driver,
            llm_client=llm_client,
            embedder=embedder,
            cross_encoder=GeminiRerankerClient(config=gemini_config),
        )

    async def _build_async():
        return _build()

    fut = asyncio.run_coroutine_threadsafe(_build_async(), loop)
    _graphiti = fut.result()
    return _graphiti


def _run(coro):
    """Submit a coroutine to Graphiti's dedicated event loop and block.

    Works uniformly from Streamlit, LangGraph nodes, and CLI. The loop lives
    for the process's lifetime; the Neo4j async driver's futures never see a
    different loop, so we never trip the 'Future attached to different loop'
    error that killed the previous ThreadPoolExecutor-per-call approach.
    """
    loop = _ensure_loop()
    fut = asyncio.run_coroutine_threadsafe(coro, loop)
    return fut.result()


def init_graph() -> None:
    g = _get_graphiti()
    _run(g.build_indices_and_constraints())


def add_claim_episode(run_id: int, city: str, statement: str,
                      source_url: str, dimension: str) -> None:
    """Ingest one verified claim as an episode. Graphiti does entity/relation
    extraction; the source URL travels in the episode description so graph
    facts remain traceable back to evidence."""
    from graphiti_core.nodes import EpisodeType

    g = _get_graphiti()
    _run(g.add_episode(
        name=f"{city}:{dimension}",
        episode_body=statement,
        source=EpisodeType.text,
        source_description=f"Verified claim from {source_url}",
        reference_time=datetime.now(timezone.utc),
        group_id=f"run-{run_id}",
    ))


def search_graph(run_id: int, query: str, limit: int = 10) -> list[dict]:
    """Graphiti hybrid search (semantic + BM25 + graph traversal) over edges."""
    g = _get_graphiti()
    results = _run(g.search(query, group_ids=[f"run-{run_id}"], num_results=limit))
    return [
        {
            "fact": r.fact,
            "source_description": getattr(r, "source_description", ""),
            "valid_at": str(getattr(r, "valid_at", "") or ""),
        }
        for r in results
    ]


def get_graph_snapshot(run_id: int, limit: int = 150) -> tuple[list[dict], list[dict]]:
    """Nodes + edges for UI visualisation, read directly via the Neo4j driver."""
    from neo4j import GraphDatabase

    s = get_settings()
    driver = GraphDatabase.driver(s.neo4j_uri, auth=(s.neo4j_user, s.neo4j_password))
    group = f"run-{run_id}"
    nodes, edges = [], []
    session_kwargs = {"database": s.neo4j_database} if s.neo4j_database else {}
    with driver.session(**session_kwargs) as session:
        recs = session.run(
            """MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity)
               WHERE a.group_id = $g AND b.group_id = $g
               RETURN a.uuid AS a_id, a.name AS a_name,
                      b.uuid AS b_id, b.name AS b_name,
                      r.fact AS fact LIMIT $limit""",
            g=group, limit=limit,
        )
        seen = set()
        for rec in recs:
            for nid, name in ((rec["a_id"], rec["a_name"]), (rec["b_id"], rec["b_name"])):
                if nid not in seen:
                    seen.add(nid)
                    nodes.append({"id": nid, "label": name})
            edges.append({"source": rec["a_id"], "target": rec["b_id"],
                          "label": (rec["fact"] or "")[:60]})
    driver.close()
    return nodes, edges

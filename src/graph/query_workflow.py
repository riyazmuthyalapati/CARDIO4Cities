"""LangGraph QUERY WORKFLOW (read path) — Agent 10, the Q&A agent.

    route -> retrieve (Graphiti + Qdrant + Postgres) -> synthesize (cited)

- Router picks retrieval emphasis: relationship questions lean on Graphiti,
  topical questions on Qdrant, meta questions ("what's missing?", "where did
  X come from?") on the Postgres ledger.
- BOTH stores contribute NUMBERED, CITABLE evidence: vector claims are [1]..[N],
  graph facts are [G1]..[Gk]. The answer prompt requires citations from both —
  the knowledge graph isn't decoration, it's a load-bearing source at query time.
- Candidates re-join Postgres by claim_id to attach verdict/scope/quote/URL.
- Synthesis is hard-constrained to the retrieved evidence; no evidence means
  an honest "our research didn't cover this" plus relevant recorded gaps.
"""
from typing import TypedDict

from langgraph.graph import END, StateGraph

from src.llm.client import invoke_json, invoke_with_fallback
from src.stores import relational, vector
from src.stores.graph import search_graph

ROUTE_SYSTEM = """Classify the user's question about a researched city."""
ROUTE_PROMPT = """Question: "{question}"

Categories:
- "relational": about people/organisations/programmes/policies and how they connect
- "topical": about facts, statistics, health indicators, situations
- "meta": about the research itself (gaps, evidence, sources, confidence)

Return ONLY JSON: {{"category": "..."}}"""

ANSWER_SYSTEM = """You are a research assistant for a City Lead. Answer ONLY
from the numbered evidence provided. TWO evidence streams are given and BOTH
are citable:

- Verified claims from source documents, numbered [1], [2], ...
- Knowledge-graph relationship facts, numbered [G1], [G2], ...

Every factual sentence must cite at least one piece of evidence like [2] or
[G3]. When you make a claim about a RELATIONSHIP (who runs X, who reports to
Y, which programme funds Z), you MUST cite a [G#] fact if one exists — that
is what the knowledge graph is for. Facts with scope=national must be
flagged as '(national data, not city-specific)'. If neither stream answers
the question, say plainly: "Our research did not surface verified
information on this" and mention any related recorded gaps. Never use
outside knowledge."""

ANSWER_PROMPT = """City: {city}
Question: {question}

NUMBERED EVIDENCE — verified claims from source documents:
{evidence}

NUMBERED KNOWLEDGE-GRAPH FACTS — entity relationships extracted from the
above claims (use these for questions about who/how connected):
{graph_facts}

RECORDED KNOWLEDGE GAPS (mention if relevant):
{gaps}

Answer concisely for a non-technical reader. Cite [n] for claim evidence
and [G#] for graph facts. Prefer [G#] citations whenever a graph fact
directly answers the question."""


class QueryState(TypedDict):
    run_id: int
    city: str
    question: str
    category: str
    evidence: list
    graph_facts: list
    gaps: list
    answer: str
    cited_claims: list


def route_node(state: QueryState) -> dict:
    try:
        raw = invoke_json(ROUTE_PROMPT.format(question=state["question"]),
                          primary="groq", system=ROUTE_SYSTEM)
        cat = raw.get("category", "topical")
    except Exception:
        cat = "topical"
    return {"category": cat if cat in ("relational", "topical", "meta") else "topical"}


def retrieve_node(state: QueryState) -> dict:
    """Retrieve from all three stores. Routing tunes retrieval WIDTH per store,
    not which store is consulted — graph facts feed synthesis for every
    question so relationships stay first-class."""
    run_id, q, category = state["run_id"], state["question"], state["category"]

    # Category-driven emphasis: relational questions pull more graph, fewer
    # vector hits; topical questions do the opposite; meta pulls a wider
    # gap-aware slice.
    if category == "relational":
        vec_limit, graph_limit = 5, 15
    elif category == "meta":
        vec_limit, graph_limit = 8, 8
    else:  # topical
        vec_limit, graph_limit = 8, 8

    # Vector: semantic claim evidence
    hits = vector.search_evidence(run_id, q, limit=vec_limit)
    claim_ids = [h["claim_id"] for h in hits if h.get("claim_id")]
    evidence = relational.get_claims(run_id, claim_ids=claim_ids or None,
                                     verified_only=True) if claim_ids else []

    # Graph: relationships — HYBRID search (semantic + BM25 + graph traversal)
    graph_facts = []
    try:
        graph_facts = search_graph(run_id, q, limit=graph_limit)
    except Exception:
        pass

    # Ledger: gaps (primary for meta questions)
    gaps = relational.get_gaps(run_id)
    if category == "meta" and not evidence:
        evidence = relational.get_claims(run_id, verified_only=True)[:12]

    return {"evidence": evidence, "graph_facts": graph_facts, "gaps": gaps}


def synthesize_node(state: QueryState) -> dict:
    ev_lines = [
        f"[{i}] (dim={e['dimension']}, scope={e['scope']}, verdict={e['verdict']}) "
        f"{e['statement']} | source: {e['source_url']}"
        for i, e in enumerate(state["evidence"], 1)
    ]
    gf_lines = [
        f"[G{i}] {f['fact']}"
        + (f" | valid_at={f['valid_at']}" if f.get("valid_at") else "")
        for i, f in enumerate(state["graph_facts"], 1)
    ]
    gap_lines = [f"- {g['dimension']}: {g['description']}" for g in state["gaps"]]

    answer = invoke_with_fallback(
        ANSWER_PROMPT.format(
            city=state["city"], question=state["question"],
            evidence="\n".join(ev_lines) or "(none retrieved)",
            graph_facts="\n".join(gf_lines) or "(none retrieved)",
            gaps="\n".join(gap_lines) or "(none recorded)",
        ),
        primary="groq", system=ANSWER_SYSTEM,
    )
    cited = [e["id"] for e in state["evidence"]]
    relational.save_chat(state["run_id"], state["question"], answer, cited)
    # Return graph_facts alongside claim citations so the UI can render both
    # kinds of evidence — the knowledge graph is a first-class source.
    return {"answer": answer, "cited_claims": state["evidence"]}


def build_query_graph():
    g = StateGraph(QueryState)
    g.add_node("route", route_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("synthesize", synthesize_node)
    g.set_entry_point("route")
    g.add_edge("route", "retrieve")
    g.add_edge("retrieve", "synthesize")
    g.add_edge("synthesize", END)
    return g.compile()


def answer_question(run_id: int, city: str, question: str) -> dict:
    graph = build_query_graph()
    result = graph.invoke({
        "run_id": run_id, "city": city, "question": question,
        "category": "", "evidence": [], "graph_facts": [], "gaps": [],
        "answer": "", "cited_claims": [],
    })
    # Expose the category and graph facts so the UI can show WHICH stores
    # answered the question and let the user inspect graph evidence.
    return {
        "answer": result["answer"],
        "cited_claims": result["cited_claims"],
        "graph_facts": result.get("graph_facts", []),
        "category": result.get("category", "topical"),
    }

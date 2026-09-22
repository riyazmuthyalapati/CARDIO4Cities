# 🫀 CARDIO4Cities — City Intelligence Engine

AI-powered live research on any city's cardiovascular health landscape. Given a city name, an orchestrated LangGraph agent workflow researches the public internet, verifies every claim independently, builds a reusable knowledge asset across **three datastores** (Postgres · Qdrant · Graphiti/Neo4j), and lets a City Lead explore, ask cited questions, and download a briefing report.

**Live app:** https://cardio4cities-riyaz.streamlit.app/

**Full design rationale:** [`ARCHITECTURE_AND_PLAN.md`](ARCHITECTURE_AND_PLAN.md)

## How the non-negotiables are met

| Requirement | Where |
|---|---|
| Live internet research | `src/agents/searcher.py` — Tavily + DuckDuckGo at request time; nothing pre-seeded |
| Orchestrated agentic workflow (LangGraph) | `src/graph/research_workflow.py` (write path), `src/graph/query_workflow.py` (read path) |
| Crawlability detection agent | `src/agents/crawl_gate.py` — robots.txt verdict **before** any fetch; evidence stored per source |
| Independent fact-checking agent with consequences | `src/agents/fact_checker.py` — different LLM family than the extractor (extractor: Gemini via aicredits; checker: Mistral via aicredits). Sees only (claim, raw source); unsupported → quarantined by graph topology, never reaches the knowledge asset or report |
| Graphiti knowledge graph, used at query time | `src/stores/graph.py` — episodes per verified claim. At query time, graph facts are numbered citable evidence (`[G1]..[Gk]`) alongside vector claims (`[1]..[N]`); the answer prompt requires `[G#]` citations for relationship questions |
| Three datastores | Supabase Postgres (audit/provenance) · Qdrant (semantic evidence) · Neo4j+Graphiti (relationships/time) |
| Evidence on every fact | Every claim: exact quote + source URL + crawl verdict + fact-check verdict; citations expandable in chat; evidence appendix in report |
| No fabrication | Extraction requires verbatim quotes; national data auto-flagged (`scope` column + UI badges); no-evidence answers say so |
| Deployed at a URL | Streamlit Community Cloud (below) |

## Setup

### 1. Provision free services (~20 min)
1. **Groq** — API key: https://console.groq.com/keys (free tier, 30 RPM; used as fallback only)
2. **Google AI Studio** — API key (Gemini + embeddings): https://aistudio.google.com/apikey
3. **Tavily** — API key: https://app.tavily.com
4. **aicredits** — paid OpenAI-compatible endpoint used by the extractor, the
   fact-checker (on a DIFFERENT model family — Mistral — for structural
   independence), and Graphiti's LLM + embeddings. Free tiers of Groq/Gemini
   cascade into 429s under parallel fan-out; aicredits sidesteps that.
5. **Supabase** — new project → Settings → Database → connection string (session pooler): https://supabase.com
6. **Qdrant Cloud** — free 1GB cluster → URL + API key: https://cloud.qdrant.io
7. **Neo4j AuraDB Free** — new instance → save URI + password: https://console.neo4j.io

### 2. Local run
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                 # fill in all keys
python scripts/bootstrap.py                          # smoke-tests all integrations
streamlit run app.py
```

### 3. Deployed instance

**Live at:** https://cardio4cities-riyaz.streamlit.app/

Hosted on Streamlit Community Cloud, wired up to Supabase (Postgres), Qdrant
Cloud, and Neo4j AuraDB Free. All secrets live in the app's Streamlit **Secrets**
panel (same keys as `.env.example`, TOML format — see
`.streamlit/secrets.toml.example`).

Before demoing:
- Open the URL a few minutes early — Streamlit Community Cloud sleeps idle apps.
- Resume the AuraDB instance in the Neo4j console (free tier auto-pauses).
- Check Tavily credit balance; DDG fallback covers exhaustion.

To deploy your own copy:
1. Push this repo to GitHub.
2. https://share.streamlit.io → New app → pick repo, main file `app.py`.
3. App → Settings → **Secrets** → paste your keys in the format of `.streamlit/secrets.toml.example`.

## Performance

Every remote round-trip on a hot path is either pooled or cached, and the
slowest post-report work runs off the critical path.

| Change | Where | Effect |
|---|---|---|
| Postgres connection pool | `src/stores/relational.py` (`psycopg_pool.ConnectionPool`) | Amortises Supabase's TLS handshake across the ~15 DB calls per run and every chat turn |
| LLM client cache | `src/llm/client.py` (`@lru_cache` on all builders) | Extract + fact_check fanouts (8 workers × ~14 sources × 2 stages) reuse one httpx client per provider — TLS keep-alive actually works |
| Shared `httpx.Client` for crawling | `src/agents/crawl_gate.py` | Robots.txt + body fetches share a connection pool with HTTP keep-alive instead of a fresh TLS handshake per URL |
| Cached query graph | `src/graph/query_workflow.py` (`@lru_cache` on `build_query_graph`) | LangGraph compile happens once per process, not per chat question |
| Cached Qdrant + Gemini embed clients | `src/stores/vector.py` | Both clients are singletons — the chat hot path stops rebuilding them per turn |
| Cached Neo4j driver for the graph tab | `src/stores/graph.py` (`_get_neo4j_driver`) | Bolt handshake to AuraDB happens once, not per tab render |
| `save_gaps` bulk insert | `src/stores/relational.py` | Single INSERT round-trip instead of N |
| **Graphiti ingest moved off the critical path** | `src/graph/research_workflow.py` (daemon thread + `get_graph_ingest_status`) | The report shows up ~15-30s sooner; the graph tab shows a live progress bar while ingest runs in the background |

**Perceived latency:** report on screen ~15-30s sooner. **Chat turn:** ~0.5-1.5s faster per question. **Full run wall-clock:** ~3-8s faster from client-reuse savings.

## Project layout
```
app.py                        Streamlit UI (research, dashboard, chat, graph, report)
src/config.py                 Settings (.env locally, st.secrets on cloud)
src/models.py                 Domain models + the 7 research dimensions
src/llm/client.py             aicredits ↔ Groq ↔ Gemini fallback client;
                              fact-checker pinned to a different family
src/agents/                   planner, searcher, crawl_gate (HTML + PDF fetch),
                              extractor, fact_checker,
                              sufficiency (judge + gap analyst), reporter
src/graph/research_workflow.py  LangGraph write path (plan→search→gate→extract→check→judge⟲→curate→report)
src/graph/query_workflow.py     LangGraph read path (route→retrieve→synthesize, cited)
src/stores/relational.py      Postgres: runs, sources, claims, gaps, chat (audit trail)
src/stores/vector.py          Qdrant: embedded evidence chunks
src/stores/graph.py           Graphiti over Neo4j: entities, relations, time
scripts/bootstrap.py          Datastore init + integration smoke test
```

## Demo-day checklist
- [ ] Open the Streamlit URL 10 min early (wakes the app)
- [ ] Resume the AuraDB instance in the Neo4j console
- [ ] Check Tavily credit balance; DDG fallback covers exhaustion
- [ ] Have one completed run loaded (sidebar) as backup while the live run executes

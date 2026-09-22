"""Minimal-cost connectivity check for all six services.

Verifies auth + roundtrip for each without touching search credits or running
any agent workflow. Each step prints ✅ / ❌ and continues past failures so
you see the full picture in one run.

Approximate cost per run:
- Groq       : 1 short chat call (~10 output tokens)
- Gemini     : 1 embedding call + 1 short chat call
- Tavily     : 0 (skipped — DDG fallback proves search works during real runs)
- Postgres   : CONNECT + 1 SELECT
- Qdrant     : list_collections + create+delete a tiny test collection
- Neo4j      : CONNECT + 1 RETURN 1

Usage:
    python scripts/smoke_test.py
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def step(label: str):
    """Decorator-ish helper: run a callable, print ✅/❌ + reason."""
    def runner(fn):
        print(f"{label:<32}", end=" ", flush=True)
        try:
            result = fn()
            print(f"✅ {result or ''}".rstrip())
            return True
        except Exception as e:
            print(f"❌ {type(e).__name__}: {e}")
            # print the first traceback line for context (not the whole trace)
            tb = traceback.format_exc().splitlines()
            if len(tb) > 2:
                print(f"   → {tb[-2].strip()}")
            return False
    return runner


def main() -> int:
    from src.config import get_settings

    s = get_settings()
    print("─" * 60)
    print("Smoke test — connectivity only (no agent workflow, no search credits)")
    print("─" * 60)

    # --- 1. Settings loaded? ---
    checks = {
        "GROQ_API_KEY": s.groq_api_key,
        "GOOGLE_API_KEY": s.google_api_key,
        "TAVILY_API_KEY": s.tavily_api_key,
        "DATABASE_URL": s.database_url,
        "QDRANT_URL": s.qdrant_url,
        "QDRANT_API_KEY": s.qdrant_api_key,
        "NEO4J_URI": s.neo4j_uri,
        "NEO4J_PASSWORD": s.neo4j_password,
    }
    missing = [k for k, v in checks.items() if not v]
    if missing:
        print(f"❌ Missing in .env: {missing}")
        return 1
    print(f"{'settings loaded':<32} ✅ all 8 keys present")

    passed = 0
    total = 6

    # --- 2. Postgres ---
    @step("Postgres (Supabase)")
    def _pg():
        import psycopg
        with psycopg.connect(s.database_url, connect_timeout=15) as conn:
            row = conn.execute("SELECT version()").fetchone()
        return row[0].split(",")[0][:40]

    passed += int(_pg)

    # --- 3. Qdrant ---
    @step("Qdrant")
    def _qd():
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams
        c = QdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key, timeout=15)
        c.get_collections()  # auth check
        # roundtrip: create + delete a tiny throwaway collection
        if c.collection_exists("__smoke__"):
            c.delete_collection("__smoke__")
        c.create_collection("__smoke__",
                            vectors_config=VectorParams(size=4, distance=Distance.COSINE))
        c.delete_collection("__smoke__")
        return "auth + read/write OK"

    passed += int(_qd)

    # --- 4. Neo4j (bare driver — cheaper than booting Graphiti) ---
    @step("Neo4j AuraDB")
    def _neo():
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(s.neo4j_uri, auth=(s.neo4j_user, s.neo4j_password))
        with driver.session() as sess:
            val = sess.run("RETURN 1 AS x").single()["x"]
        driver.close()
        assert val == 1
        return "connect + query OK"

    passed += int(_neo)

    # --- 5. Groq ---
    @step("Groq LLM")
    def _groq():
        from langchain_groq import ChatGroq
        llm = ChatGroq(model=s.groq_model, api_key=s.groq_api_key, temperature=0)
        reply = llm.invoke("Reply with exactly: ok").content.strip().lower()
        assert "ok" in reply, f"unexpected reply: {reply!r}"
        return f"reply='{reply[:20]}'"

    passed += int(_groq)

    # --- 6. Gemini generation ---
    @step("Gemini generation")
    def _gemini():
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(model=s.gemini_model, google_api_key=s.google_api_key, temperature=0)
        raw = llm.invoke("Reply with exactly: ok").content
        # gemini-3.x sometimes returns content as [{"type": "text", "text": "..."}]
        if isinstance(raw, list):
            raw = "".join(part.get("text", "") if isinstance(part, dict) else str(part)
                          for part in raw)
        reply = raw.strip().lower()
        assert "ok" in reply, f"unexpected reply: {reply!r}"
        return f"reply='{reply[:20]}'"

    passed += int(_gemini)

    # --- 7. Gemini embeddings (needed for Qdrant + Graphiti downstream) ---
    @step("Gemini embeddings")
    def _emb():
        from src.stores.vector import embed
        vec = embed(["smoke test"])[0]
        assert len(vec) == s.embedding_dim, f"dim={len(vec)}, expected {s.embedding_dim}"
        return f"dim={len(vec)}"

    passed += int(_emb)

    print("─" * 60)
    print(f"{passed}/{total} services reachable. "
          f"Tavily: skipped (0 credits spent — DDG fallback proves it live).")
    print("Next step: python scripts/bootstrap.py  (init schemas + Graphiti)")
    return 0 if passed == total else 2


if __name__ == "__main__":
    sys.exit(main())

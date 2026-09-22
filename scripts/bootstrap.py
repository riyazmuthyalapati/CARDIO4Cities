"""One-shot bootstrap + smoke test for all three datastores and both LLMs.

Run this FIRST after filling .env — it validates the riskiest integrations
(especially Graphiti + AuraDB + Gemini) before you build anything on top.

    python scripts/bootstrap.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main() -> None:
    from src.config import get_settings

    s = get_settings()
    missing = [k for k in ("groq_api_key", "google_api_key", "tavily_api_key",
                           "database_url", "qdrant_url", "neo4j_uri")
               if not getattr(s, k)]
    if missing:
        print(f"❌ Missing settings: {missing} — fill .env first")
        sys.exit(1)

    print("1/5 Postgres schema…", end=" ", flush=True)
    from src.stores import relational
    relational.init_schema()
    print("✅")

    print("2/5 Qdrant collection…", end=" ", flush=True)
    from src.stores import vector
    vector.init_collection()
    print("✅")

    print("3/5 Groq LLM…", end=" ", flush=True)
    from src.llm.client import invoke_with_fallback
    print("✅" if "ok" in invoke_with_fallback("Reply with exactly: ok").lower() else "⚠️ odd reply")

    print("4/5 Gemini embeddings…", end=" ", flush=True)
    dim = len(vector.embed(["smoke test"])[0])
    assert dim == s.embedding_dim, f"embedding dim {dim} != configured {s.embedding_dim}"
    print("✅")

    print("5/5 Graphiti + Neo4j AuraDB (slow first time)…", end=" ", flush=True)
    from src.stores.graph import init_graph
    init_graph()
    print("✅")

    print("\nAll datastores ready. Try: streamlit run app.py")


if __name__ == "__main__":
    main()

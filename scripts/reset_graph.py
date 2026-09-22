"""Drop all Graphiti data + vector indexes so Graphiti rebuilds them at the
new embedding dimension. Required when switching embedding models
(e.g. Gemini 768 -> MiniLM 384) because Neo4j binds vector dims at creation.

Safe: only touches nodes/indexes Graphiti created. Postgres and Qdrant untouched.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from neo4j import GraphDatabase
from src.config import get_settings


VECTOR_INDEX_NAMES = [
    # Graphiti's known vector-backed indexes; harmless if some don't exist.
    "entity_name_embedding_index",
    "edge_fact_embedding_index",
    "community_name_embedding_index",
    "episode_content_embedding",
]


def main() -> int:
    s = get_settings()
    d = GraphDatabase.driver(s.neo4j_uri, auth=(s.neo4j_user, s.neo4j_password))
    with d.session(database=s.neo4j_database) as ses:
        # 1. Delete all graph nodes/relationships
        result = ses.run(
            "MATCH (n) DETACH DELETE n RETURN count(n) AS deleted"
        ).single()
        print(f"  ✅ Deleted {result['deleted']} nodes (and their relationships).")

        # 2. Drop vector indexes; regular range indexes are safe to leave in place
        # but vector indexes carry a fixed dim we're changing.
        for idx in ses.run("SHOW VECTOR INDEXES YIELD name").data():
            name = idx["name"]
            try:
                ses.run(f"DROP INDEX `{name}` IF EXISTS")
                print(f"  ✅ Dropped vector index: {name}")
            except Exception as e:
                print(f"  ⚠️  Skipped {name}: {e}")
    d.close()
    print("\nDone. Next: run scripts/bootstrap.py to rebuild indexes at the new dim.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

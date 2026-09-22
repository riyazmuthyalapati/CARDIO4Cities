"""Introspect the actual Neo4j contents for a run."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from neo4j import GraphDatabase
from src.config import get_settings


def main() -> int:
    run_id = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    group = f"run-{run_id}"
    s = get_settings()
    d = GraphDatabase.driver(s.neo4j_uri, auth=(s.neo4j_user, s.neo4j_password))
    with d.session(database=s.neo4j_database) as ses:
        print(f"── Entity nodes for {group} ──")
        for r in ses.run("MATCH (n:Entity {group_id: $g}) RETURN n.name AS name, n.summary AS summary LIMIT 30",
                         g=group):
            summ = (r["summary"] or "")[:100]
            print(f"  * {r['name']}: {summ}")

        print(f"\n── Episodic nodes for {group} ──")
        for r in ses.run("MATCH (e:Episodic {group_id: $g}) RETURN e.name AS name, e.content AS content LIMIT 20",
                         g=group):
            print(f"  * {r['name']}: {(r['content'] or '')[:80]}")

        print(f"\n── All relationships in {group} ──")
        for row in ses.run("MATCH (a {group_id: $g})-[r]->(b {group_id: $g}) RETURN type(r) AS t, count(*) AS c",
                           g=group).data():
            print(f"  {row['t']}: {row['c']}")

        print(f"\n── RELATES_TO edges (Entity-Entity) for {group} ──")
        for row in ses.run(
            "MATCH (a:Entity {group_id: $g})-[r:RELATES_TO]->(b:Entity {group_id: $g}) "
            "RETURN a.name AS src, b.name AS dst, r.fact AS fact LIMIT 30", g=group):
            print(f"  {row['src']} -> {row['dst']}: {row['fact']}")
    d.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

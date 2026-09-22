"""Ingest the top verified claims from an existing run into Graphiti.

For when a run completed but the Graphiti curator failed mid-flight (e.g.
Gemini 503 outage). Reads verified claims from Postgres, does NOT touch
Postgres/Qdrant/search — only writes Graphiti episodes.

Usage:
    python scripts/repopulate_graph.py            # picks latest run
    python scripts/repopulate_graph.py 3          # specific run id
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main() -> int:
    from src.stores import relational
    from src.stores.graph import add_claim_episode

    runs = relational.list_runs()
    if not runs:
        print("No runs yet — run the app first.")
        return 1

    run_id = int(sys.argv[1]) if len(sys.argv) > 1 else runs[0]["id"]
    run = relational.get_run(run_id)
    if not run:
        print(f"Run #{run_id} not found.")
        return 1
    city = run["city"]

    verified = [c for c in relational.get_claims(run_id, verified_only=True)]
    if not verified:
        print(f"Run #{run_id} ({city}) has no verified claims.")
        return 1

    # Round-robin by dimension so the graph gets ONE fact per dimension
    # before doubling up — otherwise the top-scored dimension floods and
    # the graph collapses to 2 entities. Within a dimension, prefer
    # city-scope over national.
    SCOPE_PRIORITY = {"city": 0, "regional": 1, "national": 2, "unknown": 3}
    by_dim: dict[str, list] = {}
    for c in verified:
        by_dim.setdefault(c["dimension"], []).append(c)
    for lst in by_dim.values():
        lst.sort(key=lambda c: SCOPE_PRIORITY.get(c["scope"], 4))

    interleaved = []
    while any(by_dim.values()):
        for dim in list(by_dim.keys()):
            if by_dim[dim]:
                interleaved.append(by_dim[dim].pop(0))
    verified = interleaved
    cap = min(15, len(verified))

    print(f"─── Repopulating Graphiti for run #{run_id} ({city}) ───")
    print(f"    {len(verified)} verified claims available; ingesting top {cap}.")
    print("    Each takes ~30-60s on Gemini free tier — be patient.\n")

    import concurrent.futures as cf

    def _ingest(pair):
        i, c = pair
        t0 = time.time()
        try:
            add_claim_episode(run_id, city, c["statement"], c["source_url"],
                              c["dimension"])
            return i, time.time() - t0, None, c["statement"]
        except Exception as e:
            return i, time.time() - t0, f"{type(e).__name__}: {str(e)[:120]}", c["statement"]

    grand = time.time()
    with cf.ThreadPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(_ingest, [(i, c) for i, c in enumerate(verified[:cap], 1)]))
    ok = 0
    for i, dt, err, stmt in sorted(results):
        if err:
            print(f"  [{i}/{cap}] ❌ {dt:.1f}s — {err}")
        else:
            ok += 1
            print(f"  [{i}/{cap}] ✅ {dt:.1f}s — {stmt[:80]}")

    print(f"\n✅ Done. {ok}/{cap} episodes written to Neo4j in {time.time()-grand:.1f}s wall.")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())

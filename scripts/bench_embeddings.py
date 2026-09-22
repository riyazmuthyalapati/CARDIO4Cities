"""Bench aicredits embedding models to pick the fastest.

Measures: single-call latency (n=3) and concurrent throughput (20 parallel
calls) — the second matters more because Graphiti fires ~20 embeddings per
episode.
"""
from __future__ import annotations
import concurrent.futures as cf
import os
import sys
import time
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from openai import OpenAI

CANDIDATES = [
    "sentence-transformers/all-minilm-l6-v2",   # baseline (current)
    "sentence-transformers/all-minilm-l12-v2",
    "baai/bge-base-en-v1.5",
    "openai/text-embedding-3-small",
    "google/gemini-embedding-001",
]

SAMPLE = "The Health Insurance Institute of Slovenia funds cardiovascular disease screening in Ljubljana under the National Health Plan 2016-2025."


def bench(model: str) -> dict:
    client = OpenAI(api_key=os.environ["AICREDITS_API_KEY"],
                    base_url=os.environ["AICREDITS_BASE_URL"])
    print(f"\n── {model} ──")
    seq_times = []
    dim = 0
    err = None
    for _ in range(3):
        t = time.time()
        try:
            r = client.embeddings.create(model=model, input=SAMPLE)
            seq_times.append(time.time() - t)
            dim = len(r.data[0].embedding)
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:150]}"
            break
    if err:
        print(f"  ERR: {err}")
        return {"model": model, "error": err}
    seq_times.sort()
    print(f"  seq (n=3): p50={median(seq_times):.2f}s  min={seq_times[0]:.2f}s  max={seq_times[-1]:.2f}s  dim={dim}")

    # concurrent 20
    def _one():
        c = OpenAI(api_key=os.environ["AICREDITS_API_KEY"],
                   base_url=os.environ["AICREDITS_BASE_URL"])
        t = time.time()
        c.embeddings.create(model=model, input=SAMPLE)
        return time.time() - t

    t0 = time.time()
    par_times = []
    with cf.ThreadPoolExecutor(max_workers=10) as ex:
        futs = [ex.submit(_one) for _ in range(20)]
        for f in cf.as_completed(futs):
            try:
                par_times.append(f.result())
            except Exception as e:
                par_times.append(None)
    wall = time.time() - t0
    ok = [t for t in par_times if t is not None]
    n_err = 20 - len(ok)
    if ok:
        ok.sort()
        print(f"  par (n=20, c=10): wall={wall:.2f}s  p50={median(ok):.2f}s  p95={ok[int(len(ok)*0.95)]:.2f}s  errors={n_err}")
    else:
        print(f"  par (n=20): all errored")
    return {"model": model, "seq_p50": median(seq_times), "par_wall": wall,
            "par_p50": median(ok) if ok else None, "dim": dim, "errors": n_err}


def main() -> int:
    print("=" * 70)
    print("aicredits embedding bench — 20 concurrent calls simulate 1 Graphiti episode")
    print("=" * 70)
    results = []
    for m in CANDIDATES:
        try:
            results.append(bench(m))
        except Exception as e:
            print(f"\n{m}: fatal {type(e).__name__}: {e}")
    print("\n" + "=" * 70)
    print(f"  {'Model':<48} {'seq p50':>8} {'par wall':>9} {'par p50':>8} {'dim':>4}")
    print("=" * 70)
    for r in results:
        if "error" in r:
            print(f"  {r['model']:<48}  {r['error'][:60]}")
        else:
            print(f"  {r['model']:<48} {r['seq_p50']:>7.2f}s {r['par_wall']:>8.2f}s "
                  f"{r['par_p50']:>7.2f}s {r['dim']:>4}")
    print("\n💡 Lower par-wall = faster ep. Current dim (384=MiniLM) is baked into Neo4j.")
    print("   Switching dim requires reset_graph.py + bootstrap.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

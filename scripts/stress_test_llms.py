"""Stress-test LLM providers to see if adding GLM/Mistral would speed up
Graphiti episode ingestion (the workflow's actual bottleneck).

Measures three tasks per provider:
  1. Short completion latency  (what is 2+2)
  2. JSON extraction latency   (mimics Graphiti extract_nodes)
  3. Concurrent throughput     (5 parallel JSON extractions)

Reports: p50 / p95 latency + wall-clock for the parallel batch, so we can
tell whether swapping Graphiti's LLM would be worth the rewire.
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import sys
import time
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env into os.environ so GLM_API_KEY / MISTRAL_API_KEY are visible.
# (pydantic-settings only surfaces declared fields; ad-hoc keys stay hidden.)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

from src.config import get_settings

SHORT_PROMPT = "What is 2+2? Reply with only the number."

EXTRACT_SYSTEM = (
    "Extract entities from the text as JSON: {\"entities\": [{\"name\": str, "
    "\"type\": str}]}. Only return JSON, no prose."
)
EXTRACT_USER = (
    "The Pediatric Clinic of the University Medical Center Ljubljana in Slovenia "
    "manages cardiovascular disease screening for children with elevated cholesterol, "
    "in partnership with the Health Insurance Institute of Slovenia and NIJZ under "
    "the National Health Plan 2016–2025."
)


def _stats(times: list[float]) -> str:
    if not times:
        return "n/a"
    times = sorted(times)
    p50 = median(times)
    p95 = times[int(len(times) * 0.95)] if len(times) > 1 else times[0]
    return f"p50={p50:.2f}s  p95={p95:.2f}s  mean={sum(times)/len(times):.2f}s"


def _time(fn):
    t = time.time()
    try:
        out = fn()
        return time.time() - t, out, None
    except Exception as e:
        return time.time() - t, None, f"{type(e).__name__}: {str(e)[:120]}"


# ---------- provider callers (return callable that produces a str) ----------

def _groq(model: str, system: str, user: str):
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["GROQ_API_KEY"],
                    base_url="https://api.groq.com/openai/v1")
    def _call():
        r = client.chat.completions.create(model=model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}], temperature=0)
        return r.choices[0].message.content
    return _call


def _gemini(model: str, system: str, user: str):
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    def _call():
        r = client.models.generate_content(
            model=model,
            contents=[f"{system}\n\n{user}"],
            config=types.GenerateContentConfig(temperature=0))
        return r.text
    return _call


def _openai_compat(base_url: str, api_key: str, model: str, system: str, user: str):
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=base_url)
    def _call():
        r = client.chat.completions.create(model=model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}], temperature=0)
        return r.choices[0].message.content
    return _call


# ---------- test harness ----------

def run_provider(name: str, factory) -> dict:
    print(f"\n── {name} ──")

    # 1. Short completion
    short_times: list[float] = []
    err = None
    for _ in range(3):
        dt, out, e = _time(factory(SHORT_PROMPT.split("Reply")[0], SHORT_PROMPT))
        if e:
            err = e; break
        short_times.append(dt)
    if err:
        print(f"  short: FAILED — {err}")
        return {"name": name, "error": err}
    print(f"  short (n=3):  {_stats(short_times)}")

    # 2. JSON extraction (3 sequential)
    json_times: list[float] = []
    for _ in range(3):
        dt, out, e = _time(factory(EXTRACT_SYSTEM, EXTRACT_USER))
        if e:
            err = e; break
        json_times.append(dt)
    if err:
        print(f"  json:  FAILED — {err}")
        return {"name": name, "short": short_times, "error": err}
    print(f"  json  (n=3):  {_stats(json_times)}")

    # 3. Concurrent (5 parallel)
    t0 = time.time()
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
        futs = [ex.submit(lambda: _time(factory(EXTRACT_SYSTEM, EXTRACT_USER)))
                for _ in range(5)]
        results = [f.result() for f in futs]
    wall = time.time() - t0
    par_times = [t for t, _, e in results if not e]
    n_err = sum(1 for _, _, e in results if e)
    if n_err:
        first = next(e for _, _, e in results if e)
        print(f"  par  (n=5):   {n_err}/5 errors ({first})")
    print(f"  par  (n=5):   wall={wall:.2f}s  {_stats(par_times)}")
    return {"name": name, "short": short_times, "json": json_times,
            "par_wall": wall, "par": par_times, "par_err": n_err}


def main() -> int:
    s = get_settings()  # loads .env, configures SSL bundle
    print("=" * 70)
    print("LLM stress test — measuring providers on Graphiti-like workloads")
    print("=" * 70)

    providers = []

    # Groq — current workhorse
    if os.environ.get("GROQ_API_KEY"):
        providers.append(("Groq (openai/gpt-oss-120b)",
            lambda sysm, usrm: _groq(s.groq_model, sysm, usrm)))

    # Gemini — current Graphiti backend
    if os.environ.get("GOOGLE_API_KEY"):
        providers.append((f"Gemini ({s.gemini_model})",
            lambda sysm, usrm: _gemini(s.gemini_model, sysm, usrm)))

    # GLM (Zhipu) — free glm-4.5-flash via bigmodel.cn
    if os.environ.get("GLM_API_KEY"):
        def _mk_glm(base="https://open.bigmodel.cn/api/paas/v4",
                    key=os.environ["GLM_API_KEY"], model="glm-4.5-flash"):
            return lambda sysm, usrm: _openai_compat(base, key, model, sysm, usrm)
        providers.append(("GLM (glm-4.5-flash)", _mk_glm()))

    # Mistral — free tier is 1 req/sec strict; skip parallel and slow the sequential
    if os.environ.get("MISTRAL_API_KEY"):
        def _mk_mistral(base="https://api.mistral.ai/v1",
                        key=os.environ["MISTRAL_API_KEY"],
                        model="mistral-small-latest"):
            return lambda sysm, usrm: _openai_compat(base, key, model, sysm, usrm)
        providers.append(("Mistral (mistral-small-latest)", _mk_mistral()))

    results = []
    for name, factory in providers:
        try:
            results.append(run_provider(name, factory))
        except Exception as e:
            print(f"\n{name}: fatal {type(e).__name__}: {e}")

    # Comparison summary
    print("\n" + "=" * 70)
    print("SUMMARY — JSON extraction (Graphiti-like workload)")
    print("=" * 70)
    print(f"  {'Provider':<38} {'p50':>7} {'p95':>7} {'par-wall':>10}")
    for r in results:
        if "json" in r and r["json"]:
            js = sorted(r["json"])
            p50 = js[len(js)//2]
            p95 = js[-1]
            wall = r.get("par_wall", 0)
            print(f"  {r['name']:<38} {p50:>6.2f}s {p95:>6.2f}s {wall:>9.2f}s")
        else:
            print(f"  {r['name']:<38} {r.get('error', 'n/a')}")

    print("\n💡 Graphiti fires ~4 sequential JSON calls per episode.")
    print("   If p50 halves → episodes get ~2× faster.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

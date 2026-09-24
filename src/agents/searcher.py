"""Agent 2 — Searcher.

Runs planned queries against Tavily (primary) with DuckDuckGo fallback,
dedupes URLs, and assigns a coarse credibility tier by domain class.
"""
import re
from urllib.parse import urlparse

from src.config import get_settings
from src.models import SearchQuery, Source

# Strip `site:foo.example.com` operators (with optional quotes) so a query
# with a hallucinated portal can be retried without it.
_SITE_OP = re.compile(r'\bsite:\S+', flags=re.IGNORECASE)


def _strip_site(query: str) -> str:
    return _SITE_OP.sub("", query).strip()

# Government TLDs follow predictable patterns — this list catches them across
# countries (not just US/UK). Without it, kesehatan.jogjakota.go.id gets
# classified "unrated" and the crawl_gate ranker drops it below Wikipedia.
HIGH_CRED = (".gov", ".int", "who.int", "worldbank.org", ".edu", ".ac.",
             "un.org", "oecd.org", "nih.gov", "thelancet.com", "bmj.com",
             "nature.com", "sciencedirect.com", "ncbi.nlm.nih.gov",
             ".go.id", ".go.jp", ".go.kr", ".go.th",           # Asia (id/jp/kr/th)
             ".gob.", ".gouv.", ".admin.ch",                    # ES/PT-speaking, FR, CH
             ".gc.ca", ".govt.nz",                              # CA, NZ
             ".europa.eu")                                       # EU institutions
MEDIUM_CRED = (".org", "reuters.com", "bbc.", "statista.com")


def _credibility(domain: str) -> str:
    d = domain.lower()
    if any(m in d for m in HIGH_CRED):
        return "high"
    if any(m in d for m in MEDIUM_CRED):
        return "medium"
    return "unrated"


def _tavily_raw(query: str, max_results: int) -> list[dict]:
    from tavily import TavilyClient

    client = TavilyClient(api_key=get_settings().tavily_api_key)
    res = client.search(query=query, max_results=max_results, search_depth="basic")
    return [
        {"url": r["url"], "title": r.get("title", ""), "snippet": r.get("content", "")}
        for r in res.get("results", [])
    ]


def _ddg_raw(query: str, max_results: int) -> list[dict]:
    from ddgs import DDGS

    with DDGS() as ddgs:
        return [
            {"url": r["href"], "title": r.get("title", ""), "snippet": r.get("body", "")}
            for r in ddgs.text(query, max_results=max_results)
        ]


def _with_site_fallback(raw_fn, query: str, max_results: int) -> list[dict]:
    """Run `raw_fn(query, ...)`; if it's a `site:` query that returned zero
    hits (e.g. LLM guessed a portal that doesn't exist), retry once without
    the site: operator. Costs one extra search ONLY when the first whiffed."""
    results = raw_fn(query, max_results)
    if results or "site:" not in query.lower():
        return results
    stripped = _strip_site(query)
    if not stripped or stripped == query:
        return results
    return raw_fn(stripped, max_results)


def _tavily_search(query: str, max_results: int) -> list[dict]:
    return _with_site_fallback(_tavily_raw, query, max_results)


def _ddg_search(query: str, max_results: int) -> list[dict]:
    return _with_site_fallback(_ddg_raw, query, max_results)


def run_searches(queries: list[SearchQuery], seen_urls: set[str] | None = None) -> list[Source]:
    s = get_settings()
    seen = set(seen_urls or set())
    sources: list[Source] = []
    for q in queries:
        try:
            results = _tavily_search(q.query, s.max_results_per_query)
        except Exception:
            try:
                results = _ddg_search(q.query, s.max_results_per_query)
            except Exception:
                results = []
        for r in results:
            url = r["url"].split("#")[0]
            if url in seen:
                continue
            seen.add(url)
            domain = urlparse(url).netloc
            sources.append(Source(
                url=url, domain=domain, title=r["title"], snippet=r["snippet"],
                dimension=q.dimension, credibility_tier=_credibility(domain),
            ))
    return sources

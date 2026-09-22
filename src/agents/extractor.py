"""Agent 4 — Extractor.

Extracts atomic, evidence-linked claims from the page body that the
Crawlability Gate already fetched (see crawl_gate._fetch_body). For
snippet-only sources it falls back to the search snippet. Every claim carries
an exact quote and a geographic scope so national data can never silently pose
as city data.
"""
from src.llm.client import invoke_json
from src.models import Claim, Scope, Source

SYSTEM = """You extract atomic factual claims from source text for a
cardiovascular public-health research system. You never invent facts.
Every claim must be directly supported by a verbatim quote from the text."""

PROMPT = """City being researched: {city}
Research dimension: {dimension}
Source URL: {url}

SOURCE TEXT:
\"\"\"{text}\"\"\"

Extract up to {max_claims} atomic factual claims RELEVANT to the city and
dimension. Rules:
- One fact per claim, self-contained (resolve pronouns; name the city/entity).
- "exact_quote" must be a verbatim substring of the source text.
- "scope": does the fact describe the CITY itself, the REGION/state, or the
  NATION? Use "national" for country-level statistics even if the article
  mentions the city. Use "unknown" if unclear.
- Skip marketing, opinion and anything irrelevant to the dimension.
- If nothing relevant, return [].

Return ONLY a JSON array:
[{{"statement": "...", "exact_quote": "...", "scope": "city|regional|national|unknown"}}]"""


def extract_claims(city: str, source: Source, max_claims: int = 6) -> list[Claim]:
    # source.raw_text was populated by crawl_gate in the same concurrent wave
    # as the robots.txt fetch. Snippet-only / disallowed sources have raw_text=""
    # and fall back to the search-provider snippet.
    body = source.raw_text or source.snippet
    if not body.strip():
        return []

    try:
        # aicredits (DeepSeek) — no free-tier RPM cap. no_fallback=True so
        # that if aicredits blips, workers don't cascade onto Groq (30 RPM)
        # and trigger a rate-limit avalanche. Losing 1-2 sources per run is
        # fine; ballooning wall time from 65s to 3min is not.
        raw = invoke_json(
            PROMPT.format(city=city, dimension=source.dimension, url=source.url,
                          text=body, max_claims=max_claims),
            system=SYSTEM, no_fallback=True,
        )
    except Exception:
        return []

    claims = []
    for item in raw if isinstance(raw, list) else []:
        try:
            claims.append(Claim(
                statement=item["statement"],
                exact_quote=item.get("exact_quote", ""),
                dimension=source.dimension,
                scope=Scope(item.get("scope", "unknown")),
                source_url=source.url,
            ))
        except Exception:
            continue
    return claims

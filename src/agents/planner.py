"""Agent 1 — Research Planner.

Decomposes "understand {city}" into targeted search queries per dimension.
On later iterations it receives the Sufficiency Judge's gap feedback and
plans only for under-covered dimensions.
"""
import json

from src.config import get_settings
from src.llm.client import invoke_json
from src.models import DIMENSIONS, SearchQuery

SYSTEM = """You are the research planner for a cardiovascular public-health
intelligence system. You search THE WAY A HUMAN PUBLIC-HEALTH ANALYST DOES:
you know that the best sources for a city are almost never academic journals
or global aggregators — they're the city's OWN health department reports,
open-data portals, community-health-needs assessments (CHNA), and municipal
dashboards. You write queries that surface THOSE, not keyword bags that
Google will map to research papers."""

PROMPT = """City: {city}

Research dimensions and what they mean:
{dims}

{feedback}

Produce {n} targeted web search query(ies) per dimension. THINK LIKE A LOCAL
ANALYST who already knows where each city keeps its data. Rules:

1. Include the city name IN QUOTES in every query. Add the country if the
   city name is ambiguous (e.g. "Cambridge" UK vs US).

2. Prefer queries that name a KNOWN DOCUMENT TYPE or DATA PORTAL for the
   city. Good query building blocks (mix and match — don't include every one):
   - Health department publications: "Department of Public Health",
     "Health of the City" report, "annual health report", "chronic disease
     profile", "vital statistics".
   - Municipal open-data portals: "OpenData<City>", "<city>.gov data",
     "community health explorer", "health data portal".
   - Community assessments: "community health needs assessment", "CHNA",
     "county health rankings", "health disparities report".
   - Programme/policy filings: "<city> health strategy", "board of health",
     "chronic disease prevention plan", "resolution", "ordinance".
   - Stakeholders: "<city> hospital system", "health commissioner",
     "health department leadership".

3. AVOID academic-paper keyword bags like "hypertension diabetes dyslipidemia
   prevalence" — those return PubMed articles, not city data. If you want
   prevalence, ask for "<city> hypertension prevalence report" or
   "<city> chronic disease dashboard".

4. Vary the query verbs — some cities publish "reports", some run
   "dashboards", some post to "data portals". A good query mentions ONE
   such artifact type rather than listing every risk factor.

Return ONLY a JSON array: [{{"dimension": "...", "query": "..."}}, ...]"""


def plan_queries(city: str, gap_feedback: dict[str, str] | None = None) -> list[SearchQuery]:
    if gap_feedback:
        dims = {k: DIMENSIONS[k] for k in gap_feedback if k in DIMENSIONS}
        feedback = (
            "This is a follow-up iteration. Previous research was INSUFFICIENT "
            "for these dimensions — target the specific gaps:\n"
            + json.dumps(gap_feedback, indent=2)
        )
    else:
        dims, feedback = DIMENSIONS, ""

    dims_text = "\n".join(f"- {k}: {v}" for k, v in dims.items())
    raw = invoke_json(
        PROMPT.format(city=city, dims=dims_text, feedback=feedback,
                      n=get_settings().queries_per_dimension),
        primary="groq", system=SYSTEM,
    )
    queries = [
        SearchQuery(dimension=q["dimension"], query=q["query"])
        for q in raw
        if isinstance(q, dict) and q.get("dimension") in DIMENSIONS and q.get("query")
    ]
    # Fallback so an LLM formatting failure can't stall the whole run
    if not queries:
        queries = [
            SearchQuery(dimension=k, query=f"{city} {v.split(':')[0]}")
            for k, v in dims.items()
        ]
    return queries

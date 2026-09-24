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
intelligence system. You search THE WAY A LOCAL PUBLIC-HEALTH ANALYST DOES —
one who works IN the city's own country and knows its health information
ecosystem in the local language.

Global aggregators and academic journals are rarely the best sources for
city-level data. The best sources are usually the city's own health authority,
municipal open-data portals, and official gazettes — often published in the
country's language on a country-specific government domain."""

PROMPT = """City: {city}

Research dimensions and what they mean:
{dims}

{feedback}

Produce {n} targeted web search query(ies) per dimension.

FIRST, silently identify:
- The country this city is in.
- The language(s) that country's OFFICIAL GOVERNMENT publications appear in.
  This is NOT the city's demographic dominant language — it's the language
  the health department, gazette, and open-data portal publish in. Examples:
  Chicago = English (not Spanish, even with a large Latino population);
  Barcelona = Spanish or Catalan; Yogyakarta = Indonesian; Marseille = French;
  Zurich = German. Use the OFFICIAL PUBLICATION language, singular.
- The country's health-authority naming convention (e.g. "Department of
  Public Health" or "City Health Department" in the US, "Dinas Kesehatan"
  in Indonesia, "Agence Régionale de Santé" in France, "Secretaría de Salud"
  in Mexico, "Ministério da Saúde" / municipal "Secretaria Municipal de
  Saúde" in Brazil, "Public Health England / OHID" and "NHS" in the UK,
  "NSW Health / State Health Department" in Australia, "厚生労働省" in Japan).
- The country's official government domain suffix (.gov, .go.id, .gouv.fr,
  .gob.mx, .gov.br, .go.jp, .gc.ca, .gov.uk, .gov.au, .admin.ch, …).
- The document types cities in that country actually publish. Use the
  local terminology:
  - US: Community Health Needs Assessment (CHNA), Health of the City report,
    County Health Rankings, chronic disease profile, health equity index,
    city or county open-data portal (e.g. data.cityofchicago.org),
    Chicago Health Atlas / NYC Health Atlas / similar city health explorers.
  - UK: Joint Strategic Needs Assessment (JSNA), Public Health Outcomes
    Framework, Director of Public Health Annual Report, Fingertips profiles.
  - Australia / NZ: Primary Health Network (PHN) reports, AIHW city profile,
    state health data portal.
  - Indonesia: "profil kesehatan", "peraturan daerah", Dinkes dashboard.
  - Brazil: "boletim epidemiológico", "plano municipal de saúde", "diário
    oficial", Secretaria Municipal de Saúde reports.
  - France: ARS regional report, "santé publique France", "Plan Régional
    de Santé".
  - Etc. — apply the same reasoning for any other country.

Then write queries using rules:

1. Include the city name IN QUOTES in every query. Add the country if the
   city name is ambiguous (e.g. "Cambridge" UK vs US, "Cordoba" AR vs ES).

2. For each dimension, aim for a MIX across your {n} queries. Your
   PRIMARY lever is targeting the CITY'S OWN portals/publications in the
   OFFICIAL PUBLICATION LANGUAGE; `site:` is a bonus when applicable:
   - Every query should name at least one CITY-SPECIFIC signal — the health
     department, the open-data portal, a known artifact type, or the city's
     government domain. Bad: `"Chicago" cardiovascular disease prevalence
     mortality` (returns WHO/PubMed). Good: `"Chicago" chronic disease
     profile CDPH` or `"Chicago" Health Atlas hypertension` or `data
     cityofchicago health`.
   - Write queries in the OFFICIAL PUBLICATION LANGUAGE of the country
     (from your identification above). Do NOT switch to a demographic
     minority language even when significant (Chicago is English, not
     Spanish; Miami is English; Barcelona is Spanish/Catalan — the
     official gov language, not tourism English).
   - Use a `site:` operator ONLY when you are HIGHLY CONFIDENT the domain
     exists as you're writing it. Real, verifiable examples:
     `site:kesehatan.jogjakota.go.id`, `site:recife.pe.gov.br`,
     `site:legifrance.gouv.fr`, `site:chicago.gov`, `site:cityofboston.gov`,
     `site:london.gov.uk`, `site:health.ny.gov`. If you're guessing at a
     subdomain, DO NOT use `site:` — instead, put the country's TLD in as
     a normal keyword (e.g. `"Podgorica" zdravstvo gov.me`, `"Nuku'alofa"
     health gov.to`). A hallucinated `site:` returns zero results;
     a TLD-as-keyword still boosts government recall.
   - If the country's ecosystem is genuinely unclear to you, fall back to
     generic queries in the official language plus one query targeting
     the government TLD as a keyword — do not invent a portal.

3. Prefer queries that name a KNOWN DOCUMENT TYPE or DATA PORTAL — in the
   local terminology. Examples of building blocks (use whichever fit the
   country; do not force English terms onto non-English contexts):
   - Health authority publications: annual health report, city health
     profile, chronic disease profile, vital statistics.
   - Open-data / dashboards: municipal open-data portal, health data
     dashboard, disease surveillance.
   - Legal/policy filings: municipal ordinance, health strategy, gazette,
     resolution, regulation.
   - Programmes: screening campaign, NCD programme, hypertension programme,
     elderly-care programme (use LOCAL names — e.g. "lansia", "adulto mayor",
     "personnes âgées").

4. AVOID academic-paper keyword bags ("hypertension diabetes dyslipidemia
   prevalence") — those return PubMed articles, not city data.

5. Vary the query verbs and artifact types across your queries — don't
   repeat the same shape. One dimension × {n} queries should probe {n}
   DIFFERENT angles (e.g. dashboard, gazette, programme page).

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
        primary="aicredits", system=SYSTEM,
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

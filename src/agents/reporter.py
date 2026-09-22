"""Agent 9 — Report Generator.

Builds the downloadable markdown report from VERIFIED claims only, with
inline numeric citations, national-data flags, and a dedicated
"Gaps & Uncertainty" section. The evidence appendix maps every citation to
its quote, URL, crawl verdict and fact-check verdict.
"""
from datetime import date

from src.llm.client import invoke_with_fallback
from src.models import DIMENSIONS

SYSTEM = """You write crisp intelligence briefings for a City Lead preparing
to meet government health stakeholders. You use ONLY the numbered facts
provided. Every sentence that states a fact must end with its citation
marker(s) like [3]. Facts marked NATIONAL-LEVEL must be presented with the
caveat '(national data, not city-specific)'. Never add outside knowledge."""

PROMPT = """City: {city}
Date: {date}

NUMBERED VERIFIED FACTS (your only allowed material):
{facts}

KNOWN GAPS (include honestly):
{gaps}

Write a markdown briefing with these sections:
# {city} — City Intelligence Briefing
## Executive Summary  (5-8 sentences, cited)
{sections}
## Opportunities & Risks for CARDIO4Cities  (grounded in the cited facts; frame
gaps as open questions, do not speculate beyond the material)

Rules: every factual sentence cites [n]. If a section has no facts, write
exactly: "Our research did not surface verified city-level information on
this topic — see Gaps & Uncertainty." Do not invent anything."""

SECTION_TITLES = {
    "city_profile": "## City Profile & Governance",
    "cvd_burden": "## Cardiovascular Health Landscape",
    "health_system": "## Health System",
    "programmes": "## Existing Programmes",
    "policy": "## Policy Landscape",
    "stakeholders": "## Stakeholders & Organisations",
}


def generate_report(city: str, claims: list[dict], gaps: list[dict]) -> str:
    verified = [c for c in claims if not c["quarantined"] and c["verdict"] != "unsupported"]
    facts_lines = []
    for i, c in enumerate(verified, 1):
        flag = " [NATIONAL-LEVEL]" if c["scope"] == "national" else ""
        caveat = " [PARTIAL SUPPORT]" if c["verdict"] == "partially_supported" else ""
        facts_lines.append(f"[{i}] ({c['dimension']}){flag}{caveat} {c['statement']}")

    gaps_text = "\n".join(f"- ({g['severity']}) {g['dimension']}: {g['description']}"
                          for g in gaps) or "- none recorded"

    body = invoke_with_fallback(
        PROMPT.format(
            city=city, date=date.today().isoformat(),
            facts="\n".join(facts_lines) or "(no verified facts)",
            gaps=gaps_text,
            sections="\n".join(SECTION_TITLES.values()),
        ),
        primary="groq", system=SYSTEM,
    )

    # Deterministic sections: never let the LLM write these
    gaps_section = ["\n## Gaps & Uncertainty",
                    "*What we looked for and could not verify — stated openly "
                    "rather than guessed.*", ""]
    gaps_section += [f"- **{g['dimension']}** ({g['severity']}): {g['description']}"
                     for g in gaps] or ["- No major gaps recorded."]

    appendix = ["\n## Evidence Appendix", ""]
    for i, c in enumerate(verified, 1):
        appendix.append(
            f"**[{i}]** {c['statement']}\n"
            f"- Quote: \"{(c['exact_quote'] or '')[:300]}\"\n"
            f"- Source: {c['source_url']} (crawl: {c['crawl_verdict']}, "
            f"credibility: {c['credibility_tier']})\n"
            f"- Fact-check: {c['verdict']} — {c['checker_rationale']}\n"
        )
    quarantined_n = sum(1 for c in claims if c["quarantined"])
    appendix.append(
        f"\n*Methodology: live web research with robots.txt-gated crawling, "
        f"independent fact-checking, and quarantine of unverifiable content. "
        f"{quarantined_n} extracted claim(s) failed verification and were "
        f"excluded from this report.*"
    )
    return body + "\n" + "\n".join(gaps_section) + "\n" + "\n".join(appendix)

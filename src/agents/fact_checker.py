"""Agent 5 — Fact Checker (independent, with workflow consequences).

Independence is structural, not prompted:
- Different system prompt and provider call than the Extractor.
- Receives ONLY (claim, raw source text) — never the extractor's reasoning.
- Its verdict is written by the workflow runtime; the Curator node only ever
  receives claims that passed, so quarantine is enforced by graph edges.

Consequences:
- unsupported            -> quarantined: stored for audit, never surfaced as fact
- national_not_city      -> kept but flagged in every downstream view
- partially_supported    -> kept with caveat
"""
from src.llm.client import invoke_json
from src.models import Claim, ClaimVerdict, Scope, Source

SYSTEM = """You are an independent fact-checking auditor. You did NOT write
these claims and must not trust them. Your job is to verify each claim ONLY
against the provided source text. Be strict: if the text does not clearly
support a claim, mark it unsupported. Never use outside knowledge."""

PROMPT = """City being researched: {city}

SOURCE TEXT (the only evidence you may use):
\"\"\"{text}\"\"\"

CLAIMS to verify against that text:
{claims}

For each claim, give a verdict:
- "supported": the text clearly states this about the city
- "partially_supported": the text supports part of it, or with weaker certainty
- "unsupported": the text does not support it (or contradicts it, or the quote
  does not appear in the text)
- "national_not_city": the text supports it but the data is national/regional
  while the claim presents it as city-level

Return ONLY a JSON array, same order as the claims:
[{{"index": 0, "verdict": "...", "rationale": "one sentence"}}]"""


def check_claims(city: str, source: Source, claims: list[Claim]) -> list[Claim]:
    if not claims:
        return claims
    evidence_text = source.raw_text or source.snippet
    claims_text = "\n".join(
        f'{i}. statement: "{c.statement}" | quoted evidence: "{c.exact_quote}"'
        for i, c in enumerate(claims)
    )
    try:
        # STRUCTURAL INDEPENDENCE: fact-checker MUST use a different LLM than
        # the extractor. Extractor runs aicredits_model (Gemini family via
        # paid endpoint); the checker runs aicredits_checker_model (Mistral
        # family via the same paid endpoint). Different vendor, different
        # training, no rate-limit contention. Falls back to free Gemini only
        # if the checker model isn't configured. no_fallback=True prevents a
        # burst of 429s from cascading N workers onto the free tier at once.
        raw = invoke_json(
            PROMPT.format(city=city, text=evidence_text, claims=claims_text),
            primary="aicredits_checker", system=SYSTEM, no_fallback=True,
        )
        verdicts = {int(v["index"]): v for v in raw if isinstance(v, dict)}
    except Exception:
        verdicts = {}

    for i, claim in enumerate(claims):
        v = verdicts.get(i)
        if v is None:
            # Checker failed -> fail safe: do not surface unverified content
            claim.verdict = ClaimVerdict.UNSUPPORTED
            claim.checker_rationale = "Fact-checker unavailable; quarantined by fail-safe policy."
        else:
            try:
                claim.verdict = ClaimVerdict(v["verdict"])
            except ValueError:
                claim.verdict = ClaimVerdict.UNSUPPORTED
            claim.checker_rationale = v.get("rationale", "")

        if claim.verdict == ClaimVerdict.UNSUPPORTED:
            claim.quarantined = True
        if claim.verdict == ClaimVerdict.NATIONAL_NOT_CITY:
            claim.scope = Scope.NATIONAL
    return claims

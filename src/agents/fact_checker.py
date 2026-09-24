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
support a claim, mark it unsupported. Never use outside knowledge.

Sources may be in any language (English, Indonesian, French, Portuguese, …).
Claims are written in English but the "quoted evidence" is verbatim from the
source and stays in the source's original language. You can and should
verify multilingual sources — read the quote in its own language, check it
appears in the source text, and check the English statement is a faithful
summary of what the quote says."""

PROMPT = """City being researched: {city}

SOURCE TEXT (may be in any language — this is the only evidence you may use):
\"\"\"{text}\"\"\"

CLAIMS to verify against that text. Each claim has:
- an English "statement" (the fact as it will appear in the briefing)
- a "quoted evidence" verbatim from the source (may be in another language)

{claims}

For each claim, verify BOTH:
(a) The quoted evidence appears in the source text as a near-verbatim
    substring. Minor whitespace/newline differences from PDF extraction are
    fine; a genuinely fabricated quote is not.
(b) The English statement is a faithful, non-embellished summary of what
    the quote says. Translation is fine; adding facts not in the quote is not.

Verdicts:
- "supported": both (a) and (b) hold, and the fact is about the city itself.
- "partially_supported": (a) holds but (b) is weak — statement overstates
  or under-cites the quote.
- "unsupported": (a) fails (quote not in source, fabricated) OR (b) fails
  (statement asserts things the quote does not say).
- "national_not_city": (a) and (b) hold, but the quote describes
  national/regional data while the statement presents it as city-level.

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
        # family via the same paid endpoint). If aicredits_checker is
        # rate-limited, the client falls back to Groq (gpt-oss, a third
        # distinct family) — still independent from the extractor. Only if
        # Groq is ALSO exhausted does it hit Gemini (the extractor's family),
        # which momentarily collapses independence but is better than losing
        # a whole batch of claims to the fail-safe (which is what happened
        # in run 31 — 18/21 quarantines were rate-limit outages, not real
        # unsupported claims). Sticky cooldown in llm.client prevents N
        # workers from all discovering the rate limit independently.
        raw = invoke_json(
            PROMPT.format(city=city, text=evidence_text, claims=claims_text),
            primary="aicredits_checker", system=SYSTEM,
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

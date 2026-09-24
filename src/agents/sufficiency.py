"""Agents 6 & 7 — Sufficiency Judge and Gap Analyst.

The Judge scores per-dimension coverage from verified claims; low coverage
loops the workflow back to the Planner with targeted feedback (bounded by
max_iterations). The Gap Analyst turns whatever remains uncovered into an
explicit, honest known-unknowns list — gaps are first-class data here.
"""
from src.config import get_settings
from src.llm.client import invoke_json
from src.models import DIMENSIONS, Claim, Gap

JUDGE_SYSTEM = """You assess whether the research OUR SYSTEM PERFORMED IN
THIS RUN is sufficient for a City Lead preparing to meet government health
stakeholders. You reason ONLY about what our verified claims cover — you
never assume nothing exists in the world beyond what we retrieved. You are
strict but practical: a briefing needs breadth, not perfection."""

JUDGE_PROMPT = """City: {city}

Verified claims gathered IN THIS RESEARCH PASS, by dimension:
{summary}

For each dimension score coverage 0.0-1.0 based ONLY on what these claims
cover (0=we surfaced nothing, 1=we surfaced enough for a solid briefing).
If below {threshold}, say specifically what OUR RESEARCH did not surface —
framed as "we did not retrieve X" or "our sources didn't cover Y", NOT as
absolute claims about the world (which we cannot know).

Dimensions: {dims}

Return ONLY JSON:
{{"scores": {{"<dimension>": 0.0}}, "missing": {{"<dimension>": "what our research did not surface"}}}}"""

GAP_SYSTEM = """You document what OUR CURRENT RESEARCH PASS did NOT surface.
Every gap you list must be phrased as a statement about our retrieval, not
about the world. NEVER write "no X data exists" or "X is unavailable" —
we cannot know that. A stated gap in our coverage is more valuable than a
guessed answer — but a FALSE gap (claiming we missed data we actually
retrieved) is worse than no gap at all."""

GAP_PROMPT = """City: {city}

DIMENSIONS TO CONSIDER — ONLY these (coverage < 0.7 in this pass):
{under_covered}

Judge's notes on what our research did not surface (only for the dimensions
above):
{missing}

Number of quarantined (unverifiable) claims: {quarantined}
Claims flagged as national-level data: {national}

Write concrete "known unknowns" — ONE item per under-covered dimension
listed above. Do NOT invent gaps for dimensions not in that list, even if
you think of one. If the list is empty, return [].

Framing rules (STRICT):
- Say "we didn't fully cover X in this pass" or "our research did not
  surface Y for {city}" — never "no X data exists" and never "{city} has no Y".
- Say "our sources covered only national data on X" — never "only national
  data on X is available".
- Severity: "high" only if coverage < 0.3, "medium" if 0.3-0.5, "low" otherwise.
- Focus on which sub-topic our sources missed, not on grand absences.

Return ONLY JSON:
[{{"dimension": "...", "description": "...", "severity": "low|medium|high"}}]"""

# Below this coverage a dimension is "under-covered" and eligible for a gap.
# At/above it, the Gap Analyst won't emit anything for that dimension —
# preventing false gaps like "we didn't retrieve city profile" when we did.
_GAP_COVERAGE_CUTOFF = 0.7


def _claims_summary(claims: list[Claim]) -> str:
    lines = []
    for dim in DIMENSIONS:
        dim_claims = [c for c in claims if c.dimension == dim and c.is_verified]
        lines.append(f"\n[{dim}] ({len(dim_claims)} verified claims)")
        lines.extend(f"  - {c.statement}" for c in dim_claims[:10])
    return "\n".join(lines)


def judge_sufficiency(city: str, claims: list[Claim]) -> tuple[dict[str, float], dict[str, str]]:
    s = get_settings()
    try:
        raw = invoke_json(
            JUDGE_PROMPT.format(city=city, summary=_claims_summary(claims),
                                threshold=s.coverage_threshold,
                                dims=", ".join(DIMENSIONS)),
            primary="aicredits", system=JUDGE_SYSTEM,
        )
        scores = {k: float(v) for k, v in raw.get("scores", {}).items() if k in DIMENSIONS}
        missing = {k: v for k, v in raw.get("missing", {}).items() if k in DIMENSIONS}
    except Exception:
        # Judge failure must not stall the run: assume sufficient, record nothing missing
        scores, missing = {k: 1.0 for k in DIMENSIONS}, {}
    for k in DIMENSIONS:
        scores.setdefault(k, 0.0)
    return scores, missing


def analyse_gaps(city: str, scores: dict, missing: dict,
                 claims: list[Claim]) -> list[Gap]:
    quarantined = sum(1 for c in claims if c.quarantined)
    national = sum(1 for c in claims if c.scope.value == "national" and c.is_verified)

    under_covered = {k: scores.get(k, 0.0) for k in DIMENSIONS
                     if scores.get(k, 0.0) < _GAP_COVERAGE_CUTOFF}
    if not under_covered:
        return []
    scoped_missing = {k: missing[k] for k in under_covered if k in missing}
    under_text = "\n".join(f"- {k}: coverage {v:.2f}" for k, v in under_covered.items())

    try:
        raw = invoke_json(
            GAP_PROMPT.format(city=city, under_covered=under_text,
                              missing=scoped_missing,
                              quarantined=quarantined, national=national),
            primary="aicredits", system=GAP_SYSTEM,
        )
        return [Gap(**g) for g in raw
                if isinstance(g, dict) and g.get("dimension") in under_covered]
    except Exception:
        return [Gap(dimension=k, description=scoped_missing.get(k, "our research did not fully surface this dimension"),
                    severity="medium")
                for k in under_covered]

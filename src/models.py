"""Domain models shared across agents, stores and UI."""
from enum import Enum

from pydantic import BaseModel, Field

# The seven research dimensions = our definition of "understanding a city".
DIMENSIONS: dict[str, str] = {
    "city_profile": "Population, demographics, and governance — who is responsible for health in the city",
    "cvd_burden": "Cardiovascular disease and risk-factor burden: hypertension, type 2 diabetes, dyslipidaemia prevalence, mortality, screening rates",
    "health_system": "Primary care structure, major hospitals, health financing and insurance coverage",
    "programmes": "Existing health programmes: screening campaigns, NCD programmes, digital health initiatives",
    "policy": "National and municipal health policies, strategies and regulation relevant to cardiovascular health",
    "stakeholders": "Key people and organisations: health department, city officials, hospital networks, NGOs, academia",
    "gaps_opportunities": "Known service gaps, unmet needs, and entry points for a CVD prevention programme",
}


class CrawlVerdict(str, Enum):
    ALLOWED = "allowed"                # robots.txt permits fetching this URL
    DISALLOWED = "disallowed"          # robots.txt forbids it -> search snippet only
    SNIPPET_ONLY = "snippet_only"      # robots.txt unreachable -> conservative fallback


class ClaimVerdict(str, Enum):
    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    UNSUPPORTED = "unsupported"        # -> quarantined, never surfaced as fact
    NATIONAL_NOT_CITY = "national_not_city"  # kept, but always flagged in the UI


class Scope(str, Enum):
    CITY = "city"
    REGIONAL = "regional"
    NATIONAL = "national"
    UNKNOWN = "unknown"


class SearchQuery(BaseModel):
    dimension: str
    query: str


class Source(BaseModel):
    url: str
    domain: str = ""
    title: str = ""
    snippet: str = ""
    dimension: str = ""
    credibility_tier: str = "unrated"  # high | medium | unrated
    crawl_verdict: CrawlVerdict | None = None
    robots_evidence: str = ""
    raw_text: str = ""                 # extracted page text (empty if snippet-only)
    db_id: int | None = None


class Claim(BaseModel):
    statement: str
    exact_quote: str = ""
    dimension: str = ""
    scope: Scope = Scope.UNKNOWN
    source_url: str = ""
    verdict: ClaimVerdict | None = None
    checker_rationale: str = ""
    quarantined: bool = False
    db_id: int | None = None

    @property
    def is_verified(self) -> bool:
        return not self.quarantined and self.verdict in (
            ClaimVerdict.SUPPORTED,
            ClaimVerdict.PARTIALLY_SUPPORTED,
            ClaimVerdict.NATIONAL_NOT_CITY,
        )


class Gap(BaseModel):
    dimension: str
    description: str
    severity: str = Field(default="medium", description="low | medium | high")

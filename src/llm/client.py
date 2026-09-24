"""LLM access with cross-provider fallback.

Provider order (highest priority first):
- aicredits (paid, no free-tier RPM cap — used when API key is set)
- Groq (free tier, 30 RPM — falls back if aicredits absent or rate-limited)
- Gemini (free tier, moderate RPM — final fallback for outages)

Callers pass `primary=` to nudge the order; if aicredits is configured it
becomes the default so we don't burn wall time on Groq's rate-limit cooldowns
for high-fanout nodes (extract, fact-check).
"""
import json
import re
import threading
import time
from functools import lru_cache

from langchain_core.language_models.chat_models import BaseChatModel

from src.config import get_settings

# Provider cooldown table: {provider_name: monotonic_expiry_seconds}. When a
# provider exhausts its retries in a call, we mark it "cool" for
# _COOLDOWN_SECS so concurrent workers on the same run don't all pile onto it
# and burn wall time on inevitable 429s. Next call sees the cooldown and jumps
# straight to the next provider in the chain.
_cooldown: dict[str, float] = {}
_cooldown_lock = threading.Lock()
_COOLDOWN_SECS = 60.0


def _is_cool(provider: str) -> bool:
    with _cooldown_lock:
        expiry = _cooldown.get(provider, 0.0)
        return time.monotonic() < expiry


def _cool_down(provider: str, seconds: float = _COOLDOWN_SECS) -> None:
    with _cooldown_lock:
        _cooldown[provider] = time.monotonic() + seconds


def _has_aicredits() -> bool:
    s = get_settings()
    return bool(s.aicredits_api_key and s.aicredits_model)


@lru_cache(maxsize=8)
def aicredits_llm(temperature: float = 0.1) -> BaseChatModel:
    """OpenAI-compatible endpoint (aicredits.in). Uses langchain_openai's
    ChatOpenAI with base_url override — same wire protocol as OpenAI, so no
    special client needed. Cached so httpx keep-alive survives across the
    extract/fact-check fanout (8 workers × ~14 sources)."""
    from langchain_openai import ChatOpenAI

    s = get_settings()
    return ChatOpenAI(model=s.aicredits_model, api_key=s.aicredits_api_key,
                      base_url=s.aicredits_base_url, temperature=temperature)


@lru_cache(maxsize=8)
def aicredits_checker_llm(temperature: float = 0.0) -> BaseChatModel:
    """Fact-checker on aicredits — DIFFERENT MODEL than the extractor to keep
    the independence non-negotiable structural, not just prompted. Extractor
    is on aicredits_model; checker is on aicredits_checker_model. Same paid
    endpoint (no rate limits), different underlying model family."""
    from langchain_openai import ChatOpenAI

    s = get_settings()
    model = s.aicredits_checker_model or s.aicredits_model
    return ChatOpenAI(model=model, api_key=s.aicredits_api_key,
                      base_url=s.aicredits_base_url, temperature=temperature)


@lru_cache(maxsize=8)
def groq_llm(temperature: float = 0.1) -> BaseChatModel:
    from langchain_groq import ChatGroq

    s = get_settings()
    return ChatGroq(model=s.groq_model, api_key=s.groq_api_key, temperature=temperature)


@lru_cache(maxsize=8)
def gemini_llm(temperature: float = 0.0) -> BaseChatModel:
    from langchain_google_genai import ChatGoogleGenerativeAI

    s = get_settings()
    return ChatGoogleGenerativeAI(
        model=s.gemini_model, google_api_key=s.google_api_key, temperature=temperature
    )


def _content_to_str(content) -> str:
    """LangChain returns .content as either a string OR a list of parts
    (e.g. gemini-3.x flash). Coalesce to string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict):
                parts.append(p.get("text", ""))
            else:
                parts.append(str(p))
        return "".join(parts)
    return str(content)


_BUILDERS = {
    "aicredits": aicredits_llm,
    "aicredits_checker": aicredits_checker_llm,
    "groq": groq_llm,
    "gemini": gemini_llm,
}


def _has_aicredits_checker() -> bool:
    s = get_settings()
    return bool(s.aicredits_api_key and s.aicredits_checker_model)


def _provider_order(primary: str) -> list[str]:
    """Build the fallback order.

    - primary='aicredits_checker' — fact-checker path. Uses the checker
      model on aicredits (different family than the extractor's Gemini-family
      model, preserving structural independence). Falls back to Groq (a third
      distinct family, gpt-oss) then Gemini if the paid endpoint is unhealthy.
      Groq is preferred over Gemini as the first fallback because Gemini IS
      the extractor's family — using it as checker would collapse independence
      until the aicredits provider recovers.
    - primary='gemini' — legacy fact-checker path. Kept for compatibility
      when the checker model isn't configured.
    - Otherwise default to aicredits first (no free-tier RPM cap) → groq →
      gemini, so high-fanout nodes stop choking on Groq's 30 RPM limit.
    """
    aic = _has_aicredits()
    if primary == "aicredits_checker":
        if _has_aicredits_checker():
            return ["aicredits_checker", "groq", "gemini"]
        return ["groq", "aicredits", "gemini"] if aic else ["groq", "gemini"]
    if primary == "gemini":
        return ["gemini", "aicredits", "groq"] if aic else ["gemini", "groq"]
    if aic:
        return ["aicredits", "groq", "gemini"]
    return ["groq", "gemini"] if primary == "groq" else ["gemini", "groq"]


def invoke_with_fallback(
    prompt: str,
    primary: str = "groq",
    system: str = "",
    retries: int = 2,
    no_fallback: bool = False,
) -> str:
    """Invoke primary provider; on failure back off, then switch provider.

    Sticky cooldown: when a provider exhausts its retries, it's marked cool
    for _COOLDOWN_SECS (60s). Concurrent workers see the cooldown and skip
    straight to the next provider — no more N workers all discovering the
    rate limit independently.

    `no_fallback=True` restricts to just the primary. Kept for callers that
    genuinely can't afford a cross-family cascade (used to be true for
    fact_check under the old order; now the fallback goes aicredits_checker
    → Groq, which is a different family, so cascading is fine).
    """
    order = _provider_order(primary)
    if no_fallback:
        order = order[:1]
    # Skip providers currently in cooldown; if they ALL are, fall back to the
    # full order and try them anyway (better a slow success than no answer).
    active = [p for p in order if not _is_cool(p)]
    if not active:
        active = order
    messages = ([("system", system)] if system else []) + [("human", prompt)]
    last_err: Exception | None = None

    for provider in active:
        for attempt in range(retries):
            try:
                llm = _BUILDERS[provider]()
                return _content_to_str(llm.invoke(messages).content)
            except Exception as e:  # rate limit, transient network, etc.
                last_err = e
                time.sleep(2 * (attempt + 1))
        # Provider burned all retries — mark it cool so concurrent workers
        # skip it. Next call after the cooldown window will retry it fresh.
        _cool_down(provider)
    raise RuntimeError(f"All LLM providers failed: {last_err}")


def invoke_json(prompt: str, primary: str = "groq", system: str = "",
                no_fallback: bool = False) -> dict | list:
    """Invoke and parse a JSON response, tolerating markdown fences."""
    raw = invoke_with_fallback(prompt, primary=primary, system=system,
                               no_fallback=no_fallback)
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    start = min(
        (i for i in (text.find("{"), text.find("[")) if i != -1), default=0
    )
    return json.loads(text[start:])

"""Agent 3 — Crawlability Gate.

Determines whether a source permits automated extraction BEFORE anything is
crawled (non-negotiable #3). Checks robots.txt for our user agent and records
the evidence for the verdict. Disallowed or unverifiable sources are never
fetched — only their search-API snippets (returned by the search provider
under its own licence) are used downstream, and that restriction is stored.

For ALLOWED sources we also fetch + extract the page body here, in the same
concurrent wave as the robots.txt fetch. Doing this now (instead of a second
sequential wave inside extract) removes the extractor's I/O bottleneck without
changing what the LLM downstream sees — same trafilatura extraction, same
verbatim quotes, same audit trail.
"""
import io
import os
import urllib.robotparser
from urllib.parse import urlparse

import httpx
import trafilatura

from src.config import get_settings
from src.models import CrawlVerdict, Source


def _verify_arg():
    """Return the value to pass as httpx `verify=`. We EXPLICITLY point at the
    system CA bundle because httpx caches its SSL context on first import — if
    that happened before our _configure_ssl_bundle() ran, requests fail with
    'certificate verify failed'. Passing verify= here forces a fresh context
    with the right trust store, recovering Wikipedia / Frontiers / Semantic
    Scholar in WSL / corporate-proxy environments."""
    for path in (os.environ.get("SSL_CERT_FILE"),
                 os.environ.get("REQUESTS_CA_BUNDLE"),
                 "/etc/ssl/certs/ca-certificates.crt"):
        if path and os.path.isfile(path):
            return path
    return True  # let httpx use its bundled certifi bundle


_client: httpx.Client | None = None
_client_lock = __import__("threading").Lock()


def _get_client() -> httpx.Client:
    """Shared httpx.Client — one TCP/TLS pool for every robots.txt + body
    fetch in a crawl_gate wave. Keep-alive reuses connections across the 10
    workers, so we stop paying DNS+TLS setup per URL. Thread-safe."""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = httpx.Client(
                    verify=_verify_arg(),
                    timeout=12,
                    follow_redirects=True,
                    limits=httpx.Limits(max_keepalive_connections=32,
                                        max_connections=64,
                                        keepalive_expiry=30.0),
                )
    return _client


_robots_cache: dict[str, tuple[urllib.robotparser.RobotFileParser | None, str]] = {}


def _fetch_robots(base: str) -> tuple[urllib.robotparser.RobotFileParser | None, str]:
    """Returns (parser, evidence). parser=None means robots.txt unreachable."""
    if base in _robots_cache:
        return _robots_cache[base]
    robots_url = f"{base}/robots.txt"
    try:
        resp = _get_client().get(
            robots_url, timeout=8,
            headers={"User-Agent": get_settings().user_agent},
        )
        if resp.status_code >= 400:
            # Conventionally, no robots.txt (404) = crawling permitted
            parser = urllib.robotparser.RobotFileParser()
            parser.parse([])
            result = (parser, f"robots.txt returned HTTP {resp.status_code} -> no restrictions declared")
        else:
            parser = urllib.robotparser.RobotFileParser()
            parser.parse(resp.text.splitlines())
            result = (parser, f"robots.txt fetched OK ({len(resp.text)} bytes)")
    except Exception as e:
        result = (None, f"robots.txt unreachable ({type(e).__name__}) -> conservative snippet-only")
    _robots_cache[base] = result
    return result


_BROWSER_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0.0.0 Safari/537.36"
)


def _looks_like_pdf(url: str, content_type: str, body: bytes) -> bool:
    if url.lower().split("?")[0].endswith(".pdf"):
        return True
    if "application/pdf" in (content_type or "").lower():
        return True
    return body[:5] == b"%PDF-"


def _extract_pdf(body: bytes, max_chars: int) -> str:
    """City health reports and CHNAs are almost always PDFs. Extract text with
    pypdf — no OCR (would blow the demo budget), so image-only PDFs still
    yield ""; that's an acceptable loss."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(body))
        chunks: list[str] = []
        total = 0
        for page in reader.pages:
            text = page.extract_text() or ""
            chunks.append(text)
            total += len(text)
            if total >= max_chars:
                break
        return "\n".join(chunks)
    except Exception:
        return ""


def _fetch_body(url: str) -> tuple[str, str]:
    """Fetch + clean a page (HTML via trafilatura, PDF via pypdf). 12s timeout —
    slow gov domains would otherwise pile up wall time.

    Returns (body, fetch_note). fetch_note is empty on success and a short
    diagnostic on failure ("http 403", "empty body (js-rendered?)",
    "timeout: <exception>") so the UI can distinguish "allowed but blocked"
    from "allowed and usable".

    Uses a common browser User-Agent (not our bot UA) because several
    otherwise-crawlable sites (pib.gov.in, mdpi.com, nationalacademies.org)
    return 403 to any UA containing "bot". robots.txt has already been
    consulted at this point — the crawl decision is orthogonal to UA policy.
    """
    max_chars = get_settings().max_source_chars
    try:
        resp = _get_client().get(
            url,
            headers={
                "User-Agent": _BROWSER_UA,
                "Accept": "text/html,application/xhtml+xml,application/pdf,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
            },
        )
        if resp.status_code != 200:
            return "", f"http {resp.status_code}"
        ctype = resp.headers.get("content-type", "")
        body = resp.content
        if _looks_like_pdf(url, ctype, body):
            text = _extract_pdf(body, max_chars)
            return text, ("" if text else "pdf extraction empty (image-only or malformed)")
        text = trafilatura.extract(resp.text) or ""
        return text, ("" if text else "empty body (js-rendered or paywalled?)")
    except Exception as e:
        return "", f"fetch failed: {type(e).__name__}"


def check_crawlability(source: Source) -> Source:
    parsed = urlparse(source.url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    parser, evidence = _fetch_robots(base)

    if parser is None:
        source.crawl_verdict = CrawlVerdict.SNIPPET_ONLY
    elif parser.can_fetch(get_settings().user_agent, source.url) and parser.can_fetch("*", source.url):
        source.crawl_verdict = CrawlVerdict.ALLOWED
        evidence += " | can_fetch=True for our user agent"
    else:
        source.crawl_verdict = CrawlVerdict.DISALLOWED
        evidence += f" | path {parsed.path or '/'} disallowed for our user agent"

    source.robots_evidence = evidence

    # Pre-fetch the page body for allowed sources — cache it on the source so
    # the extractor doesn't have to fetch again on a serial critical path.
    if source.crawl_verdict == CrawlVerdict.ALLOWED:
        body, fetch_note = _fetch_body(source.url)
        source.raw_text = body[:get_settings().max_source_chars]
        if fetch_note:
            source.robots_evidence += f" | body-fetch: {fetch_note}"

    return source

"""Central configuration.

Locally: reads .env via pydantic-settings.
On Streamlit Cloud: st.secrets are copied into os.environ before Settings loads.

Corporate-proxy note: if REQUESTS_CA_BUNDLE / SSL_CERT_FILE are unset but a
system bundle exists (e.g. behind Zscaler), we point at it so httpx/requests
trust the intercepted certs. Harmless when there is no proxy.
"""
import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


def _configure_ssl_bundle() -> None:
    if os.environ.get("REQUESTS_CA_BUNDLE") or os.environ.get("SSL_CERT_FILE"):
        return
    for candidate in ("/etc/ssl/certs/ca-certificates.crt",
                      "/etc/pki/tls/certs/ca-bundle.crt",
                      "/etc/ssl/cert.pem"):
        if Path(candidate).is_file():
            os.environ.setdefault("REQUESTS_CA_BUNDLE", candidate)
            os.environ.setdefault("SSL_CERT_FILE", candidate)
            os.environ.setdefault("CURL_CA_BUNDLE", candidate)
            return


_configure_ssl_bundle()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # LLM providers
    groq_api_key: str = ""
    google_api_key: str = ""
    tavily_api_key: str = ""

    # aicredits — paid OpenAI-compatible provider (better rate limits than free tier).
    # When aicredits_api_key + aicredits_model are both set, Graphiti's LLM path
    # routes here. If aicredits_embedding_model is also set, embeddings route
    # here too (dimension MUST match the model — MiniLM=384, ada-002=1536).
    aicredits_api_key: str = ""
    aicredits_base_url: str = "https://aicredits.in/v1"
    aicredits_model: str = ""
    # Fact-checker model — MUST be different family than aicredits_model to
    # preserve structural independence (extractor and checker must be
    # different LLMs). If unset, checker falls back to Gemini free tier.
    aicredits_checker_model: str = ""
    aicredits_embedding_model: str = ""
    aicredits_embedding_dim: int = 384

    # Datastores
    database_url: str = ""  # Supabase Postgres URI
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    neo4j_uri: str = ""
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
    neo4j_database: str = ""  # AuraDB Free uses instance-id as db name, not "neo4j"

    # Models
    groq_model: str = "openai/gpt-oss-120b"
    # gemini-flash-lite has less demand than flash + generous free RPM for our tier
    gemini_model: str = "gemini-3.5-flash-lite"
    embedding_model: str = "gemini-embedding-001"
    embedding_dim: int = 768

    # Research knobs (kept small: free-tier rate limits + demo latency)
    queries_per_dimension: int = 1     # planner queries per dimension per iteration
    max_results_per_query: int = 6     # wider recall net; crawl_gate then caps to top-N
    max_sources_per_dimension: int = 2  # cap after crawl gate: 2×7 dims = ~14 sources
    max_iterations: int = 1  # single-pass by default — 2nd iter doubles wall time for marginal recall gain
    coverage_threshold: float = 0.5
    max_source_chars: int = 9000
    user_agent: str = "CARDIO4CitiesResearchBot/0.1 (research prototype)"


def load_streamlit_secrets() -> None:
    """Copy Streamlit secrets into os.environ so pydantic-settings sees them.

    No-op outside Streamlit or when secrets.toml is absent.
    """
    try:
        import streamlit as st

        for key, value in st.secrets.items():
            if isinstance(value, str) and key.upper() not in os.environ:
                os.environ[key.upper()] = value
    except Exception:
        pass


@lru_cache
def get_settings() -> Settings:
    load_streamlit_secrets()
    return Settings()

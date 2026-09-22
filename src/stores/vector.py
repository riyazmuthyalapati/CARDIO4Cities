"""Vector store (Qdrant Cloud) — semantic memory.

Stores embedded evidence passages with claim/source payloads so the Q&A agent
can find relevant verified evidence even when keywords don't match.
"""
import uuid
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, FieldCondition, Filter, MatchValue, PayloadSchemaType,
    PointStruct, VectorParams,
)

from src.config import get_settings

COLLECTION = "evidence"


@lru_cache(maxsize=1)
def get_client() -> QdrantClient:
    s = get_settings()
    return QdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key, timeout=30)


@lru_cache(maxsize=1)
def _genai_client():
    """Cache the Gemini embeddings client — same reason as get_client(): it
    holds an httpx session under the hood, no point rebuilding per embed()."""
    from google import genai
    return genai.Client(api_key=get_settings().google_api_key)


def init_collection() -> None:
    s = get_settings()
    client = get_client()
    if not client.collection_exists(COLLECTION):
        client.create_collection(
            COLLECTION,
            vectors_config=VectorParams(size=s.embedding_dim, distance=Distance.COSINE),
        )
    # Qdrant Cloud requires a payload index on any field used in a filter.
    # Idempotent: creating an existing index is a no-op.
    try:
        client.create_payload_index(
            COLLECTION, field_name="run_id",
            field_schema=PayloadSchemaType.INTEGER,
        )
    except Exception:
        pass  # already exists


_EMBED_BATCH_SIZE = 100


def embed(texts: list[str], task: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    """Gemini embeddings via the new google-genai SDK. Batched: one HTTP call
    per 100 inputs instead of one per input. Order of returned vectors matches
    the input order per the Gemini API contract."""
    from google.genai import types

    if not texts:
        return []
    s = get_settings()
    client = _genai_client()
    config = types.EmbedContentConfig(
        task_type=task, output_dimensionality=s.embedding_dim,
    )
    out: list[list[float]] = []
    for i in range(0, len(texts), _EMBED_BATCH_SIZE):
        batch = [t[:8000] for t in texts[i:i + _EMBED_BATCH_SIZE]]
        res = client.models.embed_content(
            model=s.embedding_model,
            contents=batch,
            config=config,
        )
        out.extend(list(e.values) for e in res.embeddings)
    return out


def upsert_evidence(run_id: int, items: list[dict]) -> None:
    """items: {text, claim_id, source_url, dimension, scope, verdict}"""
    if not items:
        return
    vectors = embed([i["text"] for i in items])
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vec,
            payload={"run_id": run_id, **item},
        )
        for item, vec in zip(items, vectors)
    ]
    get_client().upsert(COLLECTION, points=points)


def search_evidence(run_id: int, query: str, limit: int = 8) -> list[dict]:
    vec = embed([query], task="RETRIEVAL_QUERY")[0]
    hits = get_client().query_points(
        COLLECTION,
        query=vec,
        limit=limit,
        query_filter=Filter(must=[
            FieldCondition(key="run_id", match=MatchValue(value=run_id)),
        ]),
    ).points
    return [{**h.payload, "score": h.score} for h in hits]

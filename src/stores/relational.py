"""Relational store (Supabase Postgres) — the system of record.

Everything auditable lives here: runs, sources (with crawlability verdicts and
their robots.txt evidence), claims (with fact-check verdicts), gaps, conflicts,
and the chat log. "Where did this come from?" is a single join.
"""
import json
from contextlib import contextmanager

import psycopg
from psycopg.rows import dict_row

from src.config import get_settings
from src.models import Claim, Gap, Source

SCHEMA = """
CREATE TABLE IF NOT EXISTS research_runs (
    id SERIAL PRIMARY KEY,
    city TEXT NOT NULL,
    country TEXT DEFAULT '',
    status TEXT DEFAULT 'running',
    iterations INT DEFAULT 0,
    coverage_scores JSONB DEFAULT '{}',
    started_at TIMESTAMPTZ DEFAULT now(),
    finished_at TIMESTAMPTZ,
    report_md TEXT
);
CREATE TABLE IF NOT EXISTS sources (
    id SERIAL PRIMARY KEY,
    run_id INT REFERENCES research_runs(id),
    url TEXT NOT NULL,
    domain TEXT,
    title TEXT,
    dimension TEXT,
    credibility_tier TEXT,
    crawl_verdict TEXT,
    robots_evidence TEXT,
    fetched_at TIMESTAMPTZ DEFAULT now()
);
CREATE TABLE IF NOT EXISTS claims (
    id SERIAL PRIMARY KEY,
    run_id INT REFERENCES research_runs(id),
    source_id INT REFERENCES sources(id),
    statement TEXT NOT NULL,
    exact_quote TEXT,
    dimension TEXT,
    scope TEXT,
    verdict TEXT,
    checker_rationale TEXT,
    quarantined BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT now()
);
CREATE TABLE IF NOT EXISTS gaps (
    id SERIAL PRIMARY KEY,
    run_id INT REFERENCES research_runs(id),
    dimension TEXT,
    description TEXT,
    severity TEXT DEFAULT 'medium'
);
CREATE TABLE IF NOT EXISTS conflicts (
    id SERIAL PRIMARY KEY,
    run_id INT REFERENCES research_runs(id),
    claim_a INT REFERENCES claims(id),
    claim_b INT REFERENCES claims(id),
    description TEXT
);
CREATE TABLE IF NOT EXISTS chat_log (
    id SERIAL PRIMARY KEY,
    run_id INT REFERENCES research_runs(id),
    question TEXT,
    answer TEXT,
    cited_claim_ids JSONB DEFAULT '[]',
    created_at TIMESTAMPTZ DEFAULT now()
);
"""


@contextmanager
def get_conn():
    with psycopg.connect(get_settings().database_url, row_factory=dict_row) as conn:
        yield conn


def init_schema() -> None:
    with get_conn() as conn:
        conn.execute(SCHEMA)
        conn.commit()


def mark_stale_runs_failed(older_than_minutes: int = 30) -> int:
    """Any run still stuck in 'running' after N minutes was orphaned (crash,
    reload, network drop). Flip it to 'failed' so the sidebar dropdown stays
    honest. Returns the number of rows updated."""
    with get_conn() as conn:
        row = conn.execute(
            """UPDATE research_runs
               SET status = 'failed', finished_at = now()
               WHERE status = 'running'
                 AND started_at < now() - (%s || ' minutes')::interval
               RETURNING id""",
            (str(older_than_minutes),),
        ).fetchall()
        conn.commit()
        return len(row)


def create_run(city: str, country: str = "") -> int:
    with get_conn() as conn:
        row = conn.execute(
            "INSERT INTO research_runs (city, country) VALUES (%s, %s) RETURNING id",
            (city, country),
        ).fetchone()
        conn.commit()
        return row["id"]


def finish_run(run_id: int, status: str, iterations: int,
               coverage: dict, report_md: str = "") -> None:
    with get_conn() as conn:
        conn.execute(
            """UPDATE research_runs SET status=%s, iterations=%s,
               coverage_scores=%s, report_md=%s, finished_at=now() WHERE id=%s""",
            (status, iterations, json.dumps(coverage), report_md, run_id),
        )
        conn.commit()


def save_source(run_id: int, s: Source) -> int:
    with get_conn() as conn:
        row = conn.execute(
            """INSERT INTO sources (run_id, url, domain, title, dimension,
               credibility_tier, crawl_verdict, robots_evidence)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
            (run_id, s.url, s.domain, s.title, s.dimension, s.credibility_tier,
             s.crawl_verdict.value if s.crawl_verdict else None, s.robots_evidence),
        ).fetchone()
        conn.commit()
        return row["id"]


def save_claim(run_id: int, c: Claim, source_id: int | None) -> int:
    with get_conn() as conn:
        row = conn.execute(
            """INSERT INTO claims (run_id, source_id, statement, exact_quote,
               dimension, scope, verdict, checker_rationale, quarantined)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
            (run_id, source_id, c.statement, c.exact_quote, c.dimension,
             c.scope.value, c.verdict.value if c.verdict else None,
             c.checker_rationale, c.quarantined),
        ).fetchone()
        conn.commit()
        return row["id"]


def save_gaps(run_id: int, gaps: list[Gap]) -> None:
    with get_conn() as conn:
        for g in gaps:
            conn.execute(
                "INSERT INTO gaps (run_id, dimension, description, severity) VALUES (%s,%s,%s,%s)",
                (run_id, g.dimension, g.description, g.severity),
            )
        conn.commit()


def save_chat(run_id: int, question: str, answer: str, claim_ids: list[int]) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO chat_log (run_id, question, answer, cited_claim_ids) VALUES (%s,%s,%s,%s)",
            (run_id, question, answer, json.dumps(claim_ids)),
        )
        conn.commit()


def get_claims(run_id: int, claim_ids: list[int] | None = None,
               verified_only: bool = False) -> list[dict]:
    q = """SELECT c.*, s.url AS source_url, s.title AS source_title,
                  s.crawl_verdict, s.credibility_tier
           FROM claims c LEFT JOIN sources s ON c.source_id = s.id
           WHERE c.run_id = %s"""
    params: list = [run_id]
    if claim_ids:
        q += " AND c.id = ANY(%s)"
        params.append(claim_ids)
    if verified_only:
        q += " AND c.quarantined = FALSE AND c.verdict != 'unsupported'"
    with get_conn() as conn:
        return conn.execute(q, params).fetchall()


def get_gaps(run_id: int) -> list[dict]:
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM gaps WHERE run_id=%s ORDER BY severity", (run_id,)
        ).fetchall()


def get_run(run_id: int) -> dict | None:
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM research_runs WHERE id=%s", (run_id,)
        ).fetchone()


def list_runs() -> list[dict]:
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM research_runs ORDER BY started_at DESC LIMIT 25"
        ).fetchall()

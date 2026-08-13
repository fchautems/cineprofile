from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


DEFAULT_DB_PATH = Path(os.getenv("CINEPROFILE_DB", "data/cineprofile.db"))
SCHEMA_VERSION = 4


class CineProfileConnection(sqlite3.Connection):
    """SQLite connection whose context manager also closes the file handle.

    ``sqlite3.Connection.__exit__`` commits or rolls back but deliberately
    leaves the connection open.  That behaviour is easy to miss and prevented
    Windows from deleting audit snapshots.  Every CineProfile connection now
    has deterministic transaction *and* resource semantics.
    """

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        try:
            return bool(super().__exit__(exc_type, exc_value, traceback))
        finally:
            self.close()


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS titles (
    imdb_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    original_title TEXT,
    title_type TEXT,
    year INTEGER,
    user_rating REAL NOT NULL,
    date_rated TEXT,
    imdb_rating REAL,
    num_votes INTEGER,
    runtime_minutes INTEGER,
    release_date TEXT,
    genres_csv TEXT,
    directors_csv TEXT,
    tmdb_id INTEGER UNIQUE,
    overview TEXT,
    tagline TEXT,
    original_language TEXT,
    countries_json TEXT,
    companies_json TEXT,
    poster_path TEXT,
    tmdb_rating REAL,
    tmdb_vote_count INTEGER,
    popularity REAL,
    metadata_status TEXT NOT NULL DEFAULT 'pending',
    metadata_error TEXT,
    enriched_at TEXT
);

CREATE TABLE IF NOT EXISTS genres (
    tmdb_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS title_genres (
    imdb_id TEXT NOT NULL REFERENCES titles(imdb_id) ON DELETE CASCADE,
    genre_id INTEGER NOT NULL REFERENCES genres(tmdb_id) ON DELETE CASCADE,
    PRIMARY KEY (imdb_id, genre_id)
);

CREATE TABLE IF NOT EXISTS people (
    tmdb_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    known_for_department TEXT
);

CREATE TABLE IF NOT EXISTS credits (
    imdb_id TEXT NOT NULL REFERENCES titles(imdb_id) ON DELETE CASCADE,
    person_id INTEGER NOT NULL REFERENCES people(tmdb_id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    job TEXT NOT NULL DEFAULT '',
    character_name TEXT NOT NULL DEFAULT '',
    credit_order INTEGER,
    PRIMARY KEY (imdb_id, person_id, role, job, character_name)
);

CREATE TABLE IF NOT EXISTS keywords (
    tmdb_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS title_keywords (
    imdb_id TEXT NOT NULL REFERENCES titles(imdb_id) ON DELETE CASCADE,
    keyword_id INTEGER NOT NULL REFERENCES keywords(tmdb_id) ON DELETE CASCADE,
    PRIMARY KEY (imdb_id, keyword_id)
);

CREATE TABLE IF NOT EXISTS providers (
    imdb_id TEXT NOT NULL REFERENCES titles(imdb_id) ON DELETE CASCADE,
    region TEXT NOT NULL,
    provider_id INTEGER NOT NULL,
    provider_name TEXT NOT NULL,
    access_type TEXT NOT NULL,
    PRIMARY KEY (imdb_id, region, provider_id, access_type)
);

CREATE TABLE IF NOT EXISTS profile_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    model_version TEXT NOT NULL,
    rated_count INTEGER NOT NULL,
    enriched_count INTEGER NOT NULL,
    profile_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS recommendations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_run_id INTEGER NOT NULL REFERENCES profile_runs(id) ON DELETE CASCADE,
    tmdb_id INTEGER NOT NULL,
    imdb_id TEXT,
    title TEXT NOT NULL,
    release_date TEXT,
    score REAL NOT NULL,
    reasons_json TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS candidate_cache (
    tmdb_id INTEGER NOT NULL,
    language TEXT NOT NULL,
    region TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    PRIMARY KEY (tmdb_id, language, region)
);

CREATE TABLE IF NOT EXISTS text_embeddings (
    item_kind TEXT NOT NULL,
    item_id TEXT NOT NULL,
    model_name TEXT NOT NULL,
    text_hash TEXT NOT NULL,
    dimensions INTEGER NOT NULL,
    vector_blob BLOB NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (item_kind, item_id, model_name)
);

CREATE TABLE IF NOT EXISTS profile_preferences (
    entity_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    entity_name TEXT NOT NULL,
    adjustment INTEGER NOT NULL CHECK(adjustment BETWEEN -2 AND 2),
    updated_at TEXT NOT NULL,
    PRIMARY KEY (entity_type, entity_id)
);

CREATE TABLE IF NOT EXISTS recommendation_feedback (
    tmdb_id INTEGER PRIMARY KEY,
    imdb_id TEXT,
    title TEXT NOT NULL,
    action TEXT NOT NULL CHECK(
        action IN ('watchlist', 'not_interested', 'already_seen')
    ),
    payload_json TEXT,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS radarr_requests (
    tmdb_id INTEGER PRIMARY KEY,
    imdb_id TEXT,
    title TEXT NOT NULL,
    radarr_movie_id INTEGER,
    status TEXT NOT NULL CHECK(status IN ('downloaded')),
    payload_json TEXT,
    requested_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS radarr_request_attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tmdb_id INTEGER NOT NULL,
    imdb_id TEXT,
    title TEXT NOT NULL,
    outcome TEXT NOT NULL CHECK(
        outcome IN ('accepted', 'already_present', 'failed')
    ),
    radarr_movie_id INTEGER,
    error_message TEXT,
    attempted_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS personal_models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_version TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    trained_at TEXT NOT NULL,
    rated_count INTEGER NOT NULL,
    model_blob BLOB NOT NULL,
    metrics_json TEXT NOT NULL,
    UNIQUE(model_version, fingerprint)
);

CREATE TABLE IF NOT EXISTS active_model_configuration (
    singleton_id INTEGER PRIMARY KEY CHECK(singleton_id=1),
    engine TEXT NOT NULL,
    configuration_json TEXT NOT NULL,
    audit_created_at TEXT,
    applied_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_titles_tmdb_id ON titles(tmdb_id);
CREATE INDEX IF NOT EXISTS idx_titles_metadata_status ON titles(metadata_status);
CREATE INDEX IF NOT EXISTS idx_credits_person ON credits(person_id, role);
CREATE INDEX IF NOT EXISTS idx_title_keywords_keyword ON title_keywords(keyword_id);
CREATE INDEX IF NOT EXISTS idx_feedback_action ON recommendation_feedback(action);
CREATE INDEX IF NOT EXISTS idx_radarr_status ON radarr_requests(status);
CREATE INDEX IF NOT EXISTS idx_radarr_attempt_tmdb
    ON radarr_request_attempts(tmdb_id, attempted_at DESC);
CREATE INDEX IF NOT EXISTS idx_personal_models_version
    ON personal_models(model_version, id DESC);
"""


def db_path(path: str | Path | None = None) -> Path:
    target = Path(path) if path else DEFAULT_DB_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def connect(path: str | Path | None = None) -> sqlite3.Connection:
    connection = sqlite3.connect(
        db_path(path),
        factory=CineProfileConnection,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA busy_timeout=5000")
    return connection


def initialize(path: str | Path | None = None) -> Path:
    target = db_path(path)
    with connect(target) as connection:
        connection.executescript(SCHEMA)
        radarr_columns = {
            row["name"]
            for row in connection.execute(
                "PRAGMA table_info(radarr_requests)"
            ).fetchall()
        }
        if "payload_json" not in radarr_columns:
            connection.execute(
                "ALTER TABLE radarr_requests ADD COLUMN payload_json TEXT"
            )
        connection.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
    return target


@contextmanager
def transaction(path: str | Path | None = None) -> Iterator[sqlite3.Connection]:
    connection = connect(path)
    try:
        connection.execute("BEGIN")
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()

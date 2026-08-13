from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from .db import connect, initialize, transaction


ADJUSTMENT_LABELS = {
    -2: "Exclure",
    -1: "Réduire",
    0: "Automatique",
    1: "Favoriser",
    2: "Favoriser fortement",
}
LABEL_ADJUSTMENTS = {label: value for value, label in ADJUSTMENT_LABELS.items()}


def load_preferences(
    database: str | Path | None = None,
) -> dict[tuple[str, str], dict]:
    initialize(database)
    with connect(database) as connection:
        rows = connection.execute(
            """
            SELECT entity_type, entity_id, entity_name, adjustment
            FROM profile_preferences
            """
        ).fetchall()
    return {
        (row["entity_type"], row["entity_id"]): dict(row)
        for row in rows
    }


def save_preferences(
    rows: list[dict],
    database: str | Path | None = None,
) -> None:
    initialize(database)
    now = datetime.now(UTC).isoformat()
    with transaction(database) as connection:
        for row in rows:
            entity_type = str(row["entity_type"])
            entity_id = str(row["entity_id"])
            adjustment = int(row["adjustment"])
            if adjustment == 0 and entity_type != "interest":
                connection.execute(
                    """
                    DELETE FROM profile_preferences
                    WHERE entity_type=? AND entity_id=?
                    """,
                    (entity_type, entity_id),
                )
                continue
            connection.execute(
                """
                INSERT INTO profile_preferences(
                  entity_type, entity_id, entity_name, adjustment, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(entity_type, entity_id) DO UPDATE SET
                  entity_name=excluded.entity_name,
                  adjustment=excluded.adjustment,
                  updated_at=excluded.updated_at
                """,
                (
                    entity_type,
                    entity_id,
                    str(row["entity_name"]),
                    adjustment,
                    now,
                ),
            )


def clear_preferences(
    entity_type: str,
    database: str | Path | None = None,
) -> None:
    initialize(database)
    with transaction(database) as connection:
        connection.execute(
            "DELETE FROM profile_preferences WHERE entity_type=?",
            (str(entity_type),),
        )


def save_feedback(
    item: dict,
    action: str,
    database: str | Path | None = None,
) -> None:
    if action not in {"watchlist", "not_interested", "already_seen"}:
        raise ValueError("Retour de recommandation non reconnu.")
    initialize(database)
    with transaction(database) as connection:
        connection.execute(
            """
            INSERT INTO recommendation_feedback(
              tmdb_id, imdb_id, title, action, payload_json, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(tmdb_id) DO UPDATE SET
              imdb_id=excluded.imdb_id,
              title=excluded.title,
              action=excluded.action,
              payload_json=excluded.payload_json,
              updated_at=excluded.updated_at
            """,
            (
                int(item["tmdb_id"]),
                item.get("imdb_id"),
                item["title"],
                action,
                json.dumps(item, ensure_ascii=False),
                datetime.now(UTC).isoformat(),
            ),
        )


def remove_feedback(
    tmdb_id: int,
    database: str | Path | None = None,
) -> None:
    with transaction(database) as connection:
        connection.execute(
            "DELETE FROM recommendation_feedback WHERE tmdb_id=?",
            (int(tmdb_id),),
        )


def load_feedback(
    database: str | Path | None = None,
) -> dict[int, dict]:
    initialize(database)
    with connect(database) as connection:
        rows = connection.execute(
            """
            SELECT tmdb_id, imdb_id, title, action, payload_json, updated_at
            FROM recommendation_feedback
            """
        ).fetchall()
    result = {}
    for row in rows:
        item = dict(row)
        try:
            item["payload"] = json.loads(item.pop("payload_json") or "{}")
        except (TypeError, ValueError):
            item["payload"] = {}
        result[int(row["tmdb_id"])] = item
    return result


def load_radarr_requests(
    database: str | Path | None = None,
) -> dict[int, dict]:
    initialize(database)
    with connect(database) as connection:
        rows = connection.execute(
            """
            SELECT tmdb_id, imdb_id, title, radarr_movie_id, status,
                   payload_json, requested_at, updated_at
            FROM radarr_requests
            """
        ).fetchall()
    result = {}
    for row in rows:
        item = dict(row)
        try:
            item["payload"] = json.loads(item.pop("payload_json") or "{}")
        except (TypeError, ValueError):
            item["payload"] = {}
        result[int(row["tmdb_id"])] = item
    return result


def record_radarr_attempt(
    item: dict,
    outcome: str,
    database: str | Path | None = None,
    *,
    radarr_movie_id: int | None = None,
    error_message: str | None = None,
) -> None:
    if outcome not in {"accepted", "already_present", "failed"}:
        raise ValueError("Résultat Radarr non reconnu.")
    initialize(database)
    with transaction(database) as connection:
        connection.execute(
            """
            INSERT INTO radarr_request_attempts(
              tmdb_id, imdb_id, title, outcome, radarr_movie_id,
              error_message, attempted_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(item["tmdb_id"]),
                item.get("imdb_id"),
                item["title"],
                outcome,
                radarr_movie_id,
                error_message,
                datetime.now(UTC).isoformat(),
            ),
        )


def record_radarr_download(
    item: dict,
    radarr_movie_id: int | None,
    database: str | Path | None = None,
    *,
    already_present: bool = False,
) -> None:
    """Atomically store a successful Radarr attempt and Downloaded flag."""
    initialize(database)
    now = datetime.now(UTC).isoformat()
    with transaction(database) as connection:
        connection.execute(
            """
            INSERT INTO radarr_request_attempts(
              tmdb_id, imdb_id, title, outcome, radarr_movie_id,
              error_message, attempted_at
            ) VALUES (?, ?, ?, ?, ?, NULL, ?)
            """,
            (
                int(item["tmdb_id"]),
                item.get("imdb_id"),
                item["title"],
                "already_present" if already_present else "accepted",
                radarr_movie_id,
                now,
            ),
        )
        connection.execute(
            """
            INSERT INTO radarr_requests(
              tmdb_id, imdb_id, title, radarr_movie_id, status, payload_json,
              requested_at, updated_at
            ) VALUES (?, ?, ?, ?, 'downloaded', ?, ?, ?)
            ON CONFLICT(tmdb_id) DO UPDATE SET
              imdb_id=excluded.imdb_id,
              title=excluded.title,
              radarr_movie_id=COALESCE(
                excluded.radarr_movie_id, radarr_requests.radarr_movie_id
              ),
              status='downloaded',
              payload_json=excluded.payload_json,
              requested_at=excluded.requested_at,
              updated_at=excluded.updated_at
            """,
            (
                int(item["tmdb_id"]),
                item.get("imdb_id"),
                item["title"],
                radarr_movie_id,
                json.dumps(item, ensure_ascii=False),
                now,
                now,
            ),
        )


def load_radarr_attempts(
    database: str | Path | None = None,
    *,
    tmdb_id: int | None = None,
) -> list[dict]:
    initialize(database)
    query = """
        SELECT id, tmdb_id, imdb_id, title, outcome, radarr_movie_id,
               error_message, attempted_at
        FROM radarr_request_attempts
    """
    parameters: tuple = ()
    if tmdb_id is not None:
        query += " WHERE tmdb_id=?"
        parameters = (int(tmdb_id),)
    query += " ORDER BY attempted_at DESC, id DESC"
    with connect(database) as connection:
        rows = connection.execute(query, parameters).fetchall()
    return [dict(row) for row in rows]


def remove_radarr_request(
    tmdb_id: int,
    database: str | Path | None = None,
) -> None:
    """Remove only CineProfile's local Downloaded flag.

    Attempt history is kept and nothing is sent to Radarr.
    """
    initialize(database)
    with transaction(database) as connection:
        connection.execute(
            "DELETE FROM radarr_requests WHERE tmdb_id=?",
            (int(tmdb_id),),
        )

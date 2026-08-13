"""Durable storage for the last recommendation selection of a profile run."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .db import connect, initialize, transaction


def save_recommendation_state(
    profile_run_id: int | None,
    settings: dict[str, Any],
    diagnostics: dict[str, Any],
    database: str | Path | None = None,
) -> None:
    """Save the context belonging to the recommendation rows of one profile."""
    if not profile_run_id:
        return
    initialize(database)
    with transaction(database) as connection:
        connection.execute(
            """
            INSERT INTO recommendation_states(
              profile_run_id, settings_json, diagnostics_json, updated_at
            ) VALUES (?, ?, ?, ?)
            ON CONFLICT(profile_run_id) DO UPDATE SET
              settings_json=excluded.settings_json,
              diagnostics_json=excluded.diagnostics_json,
              updated_at=excluded.updated_at
            """,
            (
                int(profile_run_id),
                json.dumps(settings, ensure_ascii=False),
                json.dumps(diagnostics, ensure_ascii=False),
                datetime.now(UTC).isoformat(),
            ),
        )


def _json_object(value: str | None) -> dict[str, Any]:
    try:
        payload = json.loads(value or "{}")
    except (TypeError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def load_saved_recommendations(
    profile_run_id: int | None,
    database: str | Path | None = None,
) -> dict[str, Any] | None:
    """Return the latest saved selection for the current profile, if any.

    Recommendation rows already existed before version 0.13.  They can still
    be restored even if no companion state row has been written yet.
    """
    if not profile_run_id:
        return None
    initialize(database)
    with connect(database) as connection:
        rows = connection.execute(
            """
            SELECT payload_json, created_at
            FROM recommendations
            WHERE profile_run_id=?
            ORDER BY id ASC
            """,
            (int(profile_run_id),),
        ).fetchall()
        state = connection.execute(
            """
            SELECT settings_json, diagnostics_json, updated_at
            FROM recommendation_states
            WHERE profile_run_id=?
            """,
            (int(profile_run_id),),
        ).fetchone()
    if not rows:
        return None

    recommendations: list[dict] = []
    for row in rows:
        item = _json_object(row["payload_json"])
        if item:
            recommendations.append(item)
    if not recommendations:
        return None
    return {
        "recommendations": recommendations,
        "settings": _json_object(state["settings_json"]) if state else {},
        "diagnostics": _json_object(state["diagnostics_json"]) if state else {},
        "updated_at": str(
            state["updated_at"] if state else rows[-1]["created_at"]
        ),
    }

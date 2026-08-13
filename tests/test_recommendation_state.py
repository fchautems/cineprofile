from __future__ import annotations

import json
from pathlib import Path

from cineprofile.db import initialize, transaction
from cineprofile.recommendation_state import (
    load_saved_recommendations,
    save_recommendation_state,
)


def test_saved_recommendations_restore_rows_and_context(tmp_path: Path) -> None:
    database = tmp_path / "cineprofile.db"
    initialize(database)
    with transaction(database) as connection:
        profile_run_id = connection.execute(
            """
            INSERT INTO profile_runs(
              created_at, model_version, rated_count, enriched_count, profile_json
            ) VALUES ('2026-08-13T10:00:00+00:00', 'test', 1, 1, '{}')
            """
        ).lastrowid
        connection.execute(
            """
            INSERT INTO recommendations(
              profile_run_id, tmdb_id, imdb_id, title, release_date, score,
              reasons_json, payload_json, created_at
            ) VALUES (?, 123, NULL, 'Film test', '2026-01-01', 75, '[]', ?,
                      '2026-08-13T10:01:00+00:00')
            """,
            (profile_run_id, json.dumps({"tmdb_id": 123, "title": "Film test"})),
        )

    save_recommendation_state(
        profile_run_id,
        {"depth": "Normale"},
        {"search_id": "search-test"},
        database,
    )
    restored = load_saved_recommendations(profile_run_id, database)

    assert restored is not None
    assert restored["recommendations"] == [{"tmdb_id": 123, "title": "Film test"}]
    assert restored["settings"] == {"depth": "Normale"}
    assert restored["diagnostics"] == {"search_id": "search-test"}


def test_saved_recommendations_can_restore_pre_013_rows(tmp_path: Path) -> None:
    database = tmp_path / "cineprofile.db"
    initialize(database)
    with transaction(database) as connection:
        profile_run_id = connection.execute(
            """
            INSERT INTO profile_runs(
              created_at, model_version, rated_count, enriched_count, profile_json
            ) VALUES ('2026-08-13T10:00:00+00:00', 'test', 1, 1, '{}')
            """
        ).lastrowid
        connection.execute(
            """
            INSERT INTO recommendations(
              profile_run_id, tmdb_id, imdb_id, title, release_date, score,
              reasons_json, payload_json, created_at
            ) VALUES (?, 124, NULL, 'Ancien film', NULL, 70, '[]', ?,
                      '2026-08-13T10:01:00+00:00')
            """,
            (profile_run_id, json.dumps({"tmdb_id": 124, "title": "Ancien film"})),
        )

    restored = load_saved_recommendations(profile_run_id, database)

    assert restored is not None
    assert restored["settings"] == {}
    assert restored["diagnostics"] == {}
    assert restored["recommendations"][0]["title"] == "Ancien film"

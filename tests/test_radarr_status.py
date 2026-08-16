from __future__ import annotations

from cineprofile.radarr_status import (
    active_radarr_ids,
    radarr_status_meta,
    without_active_radarr_movies,
)


def test_active_radarr_movies_are_removed_from_suggestions() -> None:
    recommendations = [
        {"tmdb_id": 1, "title": "Surveillé"},
        {"tmdb_id": 2, "title": "Erreur"},
        {"tmdb_id": 3, "title": "Téléchargé"},
    ]
    requests = {
        1: {"radarr_state": "monitored"},
        2: {"radarr_state": "error"},
        3: {"radarr_state": "downloaded"},
    }

    assert active_radarr_ids(requests) == {1, 3}
    assert [
        row["tmdb_id"]
        for row in without_active_radarr_movies(recommendations, requests)
    ] == [2]


def test_radarr_badges_use_the_product_status_palette() -> None:
    assert radarr_status_meta("monitored") == (
        "violet",
        ":material/radar:",
        "Surveillé",
    )
    assert radarr_status_meta("downloading")[0] == "orange"
    assert radarr_status_meta("available")[0] == "blue"
    assert radarr_status_meta("downloaded")[0] == "green"
    assert radarr_status_meta("error")[0] == "red"
    assert radarr_status_meta("missing")[0] == "gray"

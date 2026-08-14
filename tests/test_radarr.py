from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx

from cineprofile.preferences import (
    load_radarr_attempts,
    load_radarr_requests,
    record_radarr_download,
    remove_radarr_request,
    update_radarr_states,
)
from cineprofile.radarr import RadarrClient, RadarrError
from cineprofile.radarr_sync import radarr_states_stale
from cineprofile.settings import (
    forget_radarr_connection_file,
    save_radarr_connection_file,
)


def test_radarr_adds_movie_and_starts_search() -> None:
    received_movie: dict = {}
    received_command: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["X-Api-Key"] == "secret"
        if request.url.path == "/api/v3/movie" and request.method == "GET":
            return httpx.Response(200, json=[])
        if request.url.path == "/api/v3/movie/lookup":
            assert request.url.params["term"] == "tmdb:123"
            return httpx.Response(
                200,
                json=[{"tmdbId": 123, "title": "Film test", "year": 2026}],
            )
        if request.url.path == "/api/v3/movie" and request.method == "POST":
            received_movie.update(json.loads(request.content))
            return httpx.Response(201, json={"id": 77, "tmdbId": 123})
        if request.url.path == "/api/v3/command" and request.method == "POST":
            received_command.update(json.loads(request.content))
            return httpx.Response(
                201,
                json={"id": 501, "name": "MoviesSearch", "status": "queued"},
            )
        raise AssertionError(f"Requête inattendue : {request.method} {request.url}")

    with RadarrClient(
        "http://radarr.local:7878/",
        "secret",
        transport=httpx.MockTransport(handler),
    ) as client:
        result = client.add_movie(
            123,
            root_folder_path="/movies",
            quality_profile_id=4,
        )

    assert result.movie_id == 77
    assert result.search_command_id == 501
    assert not result.already_present
    assert received_movie["tmdbId"] == 123
    assert received_movie["rootFolderPath"] == "/movies"
    assert received_movie["qualityProfileId"] == 4
    assert received_movie["monitored"] is True
    assert received_movie["addOptions"] == {
        "searchForMovie": False,
        "monitor": "movieOnly",
    }
    assert received_command == {"name": "MoviesSearch", "movieIds": [77]}


def test_radarr_existing_movie_is_not_added_twice() -> None:
    calls: list[tuple[str, str]] = []
    received_command: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        if request.url.path == "/api/v3/movie":
            return httpx.Response(200, json=[{"id": 88, "tmdbId": 123}])
        if request.url.path == "/api/v3/command":
            received_command.update(json.loads(request.content))
            return httpx.Response(201, json={"id": 502, "status": "queued"})
        raise AssertionError(f"Requête inattendue : {request.method} {request.url}")

    with RadarrClient(
        "http://radarr.local:7878",
        "secret",
        transport=httpx.MockTransport(handler),
    ) as client:
        result = client.add_movie(
            123,
            root_folder_path="/movies",
            quality_profile_id=4,
        )

    assert result.movie_id == 88
    assert result.search_command_id == 502
    assert result.already_present
    assert calls == [
        ("GET", "/api/v3/movie"),
        ("POST", "/api/v3/command"),
    ]
    assert received_command == {"name": "MoviesSearch", "movieIds": [88]}


def test_radarr_existing_unmonitored_movie_is_reactivated() -> None:
    monitored_payload: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v3/movie" and request.method == "GET":
            return httpx.Response(
                200,
                json=[
                    {
                        "id": 88,
                        "tmdbId": 123,
                        "title": "Film",
                        "monitored": False,
                    }
                ],
            )
        if request.url.path == "/api/v3/movie/88" and request.method == "PUT":
            monitored_payload.update(json.loads(request.content))
            return httpx.Response(200, json=monitored_payload)
        if request.url.path == "/api/v3/command":
            return httpx.Response(201, json={"id": 503, "status": "queued"})
        raise AssertionError(f"Requête inattendue : {request.method} {request.url}")

    with RadarrClient(
        "http://radarr.local:7878",
        "secret",
        transport=httpx.MockTransport(handler),
    ) as client:
        result = client.add_movie(
            123,
            root_folder_path="/movies",
            quality_profile_id=4,
        )

    assert result.already_present
    assert monitored_payload["monitored"] is True


def test_radarr_does_not_report_success_when_search_is_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v3/movie" and request.method == "GET":
            return httpx.Response(200, json=[])
        if request.url.path == "/api/v3/movie/lookup":
            return httpx.Response(200, json=[{"tmdbId": 123, "title": "Film"}])
        if request.url.path == "/api/v3/movie" and request.method == "POST":
            return httpx.Response(201, json={"id": 77, "tmdbId": 123})
        if request.url.path == "/api/v3/command":
            return httpx.Response(400, text="No indexer available")
        raise AssertionError(f"Requête inattendue : {request.method} {request.url}")

    with RadarrClient(
        "http://radarr.local:7878",
        "secret",
        transport=httpx.MockTransport(handler),
    ) as client:
        try:
            client.add_movie(
                123,
                root_folder_path="/movies",
                quality_profile_id=4,
            )
        except RadarrError as exc:
            message = str(exc)
        else:
            raise AssertionError("RadarrError attendu")

    assert "film est bien dans Radarr" in message
    assert "No indexer available" in message


def test_radarr_exposes_real_movie_states() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v3/movie":
            return httpx.Response(
                200,
                json=[
                    {
                        "id": 1,
                        "tmdbId": 101,
                        "hasFile": True,
                        "monitored": True,
                        "sizeOnDisk": 2 * 1024**3,
                        "movieFile": {"relativePath": "Film.mkv"},
                    },
                    {"id": 2, "tmdbId": 102, "hasFile": False, "monitored": True},
                    {
                        "id": 3,
                        "tmdbId": 103,
                        "hasFile": False,
                        "monitored": True,
                        "lastSearchTime": "2026-08-14T10:00:00Z",
                    },
                    {"id": 4, "tmdbId": 104, "hasFile": False, "monitored": True},
                    {"id": 5, "tmdbId": 105, "hasFile": False, "monitored": False},
                ],
            )
        if request.url.path == "/api/v3/queue":
            assert request.url.params["includeMovie"] == "true"
            return httpx.Response(
                200,
                json={
                    "records": [
                        {
                            "movieId": 2,
                            "trackedDownloadStatus": "ok",
                            "trackedDownloadState": "downloading",
                            "size": 100,
                            "sizeleft": 25,
                        },
                        {
                            "movieId": 4,
                            "trackedDownloadStatus": "error",
                            "trackedDownloadState": "failed",
                            "errorMessage": "Téléchargement en échec",
                        },
                    ]
                },
            )
        raise AssertionError(f"Requête inattendue : {request.method} {request.url}")

    with RadarrClient(
        "http://radarr.local:7878",
        "secret",
        transport=httpx.MockTransport(handler),
    ) as client:
        states = client.movie_states({101, 102, 103, 104, 105, 106})

    assert states[101]["state"] == "available"
    assert states[101]["progress"] == 100.0
    assert states[102]["state"] == "downloading"
    assert states[102]["progress"] == 75.0
    assert states[103]["state"] == "no_result"
    assert states[104]["state"] == "error"
    assert "échec" in states[104]["detail"]
    assert states[105]["state"] == "unmonitored"
    assert states[106]["state"] == "missing"


def test_radarr_request_status_is_persistent_and_separate(tmp_path: Path) -> None:
    database = tmp_path / "cineprofile.db"
    item = {
        "tmdb_id": 123,
        "imdb_id": "tt1234567",
        "title": "Film test",
    }

    record_radarr_download(item, 77, database)
    status = load_radarr_requests(database)[123]

    assert status["status"] == "downloaded"
    assert status["radarr_state"] == "sent"
    assert status["radarr_movie_id"] == 77
    assert status["payload"]["title"] == "Film test"
    assert load_radarr_attempts(database, tmdb_id=123)[0]["outcome"] == "accepted"

    update_radarr_states(
        {
            123: {
                "state": "available",
                "detail": "Film.mkv",
                "progress": 100.0,
                "radarr_movie_id": 77,
            }
        },
        database,
    )
    synchronized = load_radarr_requests(database)[123]
    assert synchronized["radarr_state"] == "available"
    assert synchronized["status_detail"] == "Film.mkv"
    assert synchronized["progress"] == 100.0
    assert synchronized["status_checked_at"]

    remove_radarr_request(123, database)
    assert 123 not in load_radarr_requests(database)
    assert len(load_radarr_attempts(database, tmdb_id=123)) == 1


def test_radarr_state_refresh_does_not_run_on_every_ui_click() -> None:
    now = datetime(2026, 8, 14, 12, 0, tzinfo=UTC)
    fresh = {1: {"status_checked_at": (now - timedelta(seconds=5)).isoformat()}}
    stale = {1: {"status_checked_at": (now - timedelta(seconds=40)).isoformat()}}

    assert not radarr_states_stale(fresh, now=now)
    assert radarr_states_stale(stale, now=now)
    assert radarr_states_stale({1: {"status_checked_at": None}}, now=now)


def test_radarr_credentials_can_be_saved_and_forgotten(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"

    save_radarr_connection_file(
        env_file,
        "http://radarr.local:7878/",
        "secret",
    )
    saved = env_file.read_text(encoding="utf-8")
    assert 'RADARR_URL="http://radarr.local:7878"' in saved
    assert 'RADARR_API_KEY="secret"' in saved

    forget_radarr_connection_file(env_file)
    forgotten = env_file.read_text(encoding="utf-8")
    assert "RADARR_URL" not in forgotten
    assert "RADARR_API_KEY" not in forgotten


def test_schema_migrates_an_early_radarr_table(tmp_path: Path) -> None:
    from cineprofile.db import connect, initialize

    database = tmp_path / "old-v12.db"
    connection = sqlite3.connect(database)
    connection.execute(
        """
        CREATE TABLE radarr_requests (
          tmdb_id INTEGER PRIMARY KEY,
          imdb_id TEXT,
          title TEXT NOT NULL,
          radarr_movie_id INTEGER,
          status TEXT NOT NULL,
          requested_at TEXT NOT NULL,
          updated_at TEXT NOT NULL
        )
        """
    )
    connection.commit()
    connection.close()

    initialize(database)

    with connect(database) as migrated:
        columns = {
            row["name"]
            for row in migrated.execute(
                "PRAGMA table_info(radarr_requests)"
            ).fetchall()
        }
        attempts_table = migrated.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='radarr_request_attempts'
            """
        ).fetchone()
    assert "payload_json" in columns
    assert {
        "radarr_state",
        "status_detail",
        "progress",
        "status_checked_at",
    } <= columns
    assert attempts_table is not None

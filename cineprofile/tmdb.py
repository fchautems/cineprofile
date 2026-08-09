from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Callable, Iterable

import httpx

from . import __version__
from .db import connect, initialize, transaction
from .media_types import tmdb_media_type


API_BASE = "https://api.themoviedb.org/3"


class TmdbError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        path: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.path = path


@dataclass(frozen=True)
class EnrichmentProgress:
    processed: int
    total: int
    imdb_id: str
    title: str
    status: str


class TmdbClient:
    def __init__(
        self,
        token: str,
        language: str = "fr-FR",
        region: str = "CH",
        timeout: float = 30.0,
    ) -> None:
        if not token.strip():
            raise ValueError("Le jeton TMDB est vide.")
        self.language = language
        self.region = region
        self.client = httpx.Client(
            base_url=API_BASE,
            headers={
                "Authorization": f"Bearer {token.strip()}",
                "accept": "application/json",
                "User-Agent": f"CineProfile/{__version__}",
            },
            timeout=timeout,
        )

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "TmdbClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def get(self, path: str, **params: object) -> dict:
        params.setdefault("language", self.language)
        last_network_error: httpx.RequestError | None = None
        for attempt in range(5):
            try:
                response = self.client.get(path, params=params)
            except httpx.RequestError as exc:
                last_network_error = exc
                if attempt < 4:
                    time.sleep(0.5 * (2**attempt))
                continue
            if response.status_code == 429:
                try:
                    retry_after = float(
                        response.headers.get("Retry-After", 1.0)
                    )
                except (TypeError, ValueError):
                    retry_after = 1.0
                if attempt < 4:
                    time.sleep(max(0.0, min(retry_after, 10.0)))
                continue
            if response.status_code >= 500:
                if attempt < 4:
                    time.sleep(0.5 * (2**attempt))
                continue
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                detail = response.text[:500]
                raise TmdbError(
                    f"TMDB {response.status_code} sur {path}: {detail}",
                    status_code=response.status_code,
                    path=path,
                ) from exc
            return response.json()
        detail = (
            f" Dernière erreur réseau : {last_network_error}"
            if last_network_error is not None
            else ""
        )
        raise TmdbError(
            f"TMDB ne répond pas après plusieurs tentatives sur {path}.{detail}",
            path=path,
        ) from last_network_error

    def resolve_imdb_id(self, imdb_id: str) -> tuple[str, int] | None:
        data = self.get(
            f"/find/{imdb_id}",
            external_source="imdb_id",
        )
        if data.get("movie_results"):
            return "movie", int(data["movie_results"][0]["id"])
        if data.get("tv_results"):
            return "tv", int(data["tv_results"][0]["id"])
        return None

    def details(self, media_type: str, tmdb_id: int) -> dict:
        if media_type not in {"movie", "tv"}:
            raise ValueError(f"Type TMDB non pris en charge : {media_type}")
        append = "credits,keywords,watch/providers,external_ids"
        return self.get(
            f"/{media_type}/{tmdb_id}",
            append_to_response=append,
        )

    def discover_recent_movies(
        self,
        start_date: str | None,
        end_date: str | None,
        pages: int = 5,
        min_votes: int = 0,
        *,
        with_people: int | None = None,
        with_genres: int | None = None,
        with_keywords: int | None = None,
        sort_by: str = "popularity.desc",
    ) -> list[dict]:
        results: dict[int, dict] = {}
        window_start = date.fromisoformat(start_date) if start_date else None
        window_end = date.fromisoformat(end_date) if end_date else None
        for page in range(1, max(1, pages) + 1):
            parameters: dict[str, object] = {
                "region": self.region,
                "sort_by": sort_by,
                "include_adult": "false",
                "include_video": "false",
                "vote_count.gte": min_votes,
                "page": page,
            }
            if start_date:
                parameters["primary_release_date.gte"] = start_date
            if end_date:
                parameters["primary_release_date.lte"] = end_date
            if with_people:
                parameters["with_people"] = int(with_people)
            if with_genres:
                parameters["with_genres"] = int(with_genres)
            if with_keywords:
                parameters["with_keywords"] = int(with_keywords)
            payload = self.get(
                "/discover/movie",
                **parameters,
            )
            for item in payload.get("results", []):
                if window_start or window_end:
                    try:
                        released = date.fromisoformat(
                            item.get("release_date", "")
                        )
                    except ValueError:
                        continue
                    if window_start and released < window_start:
                        continue
                    if window_end and released > window_end:
                        continue
                results[int(item["id"])] = item
            if page >= int(payload.get("total_pages", page)):
                break
        return list(results.values())

    def movie_recommendations(
        self,
        tmdb_id: int,
        *,
        pages: int = 1,
    ) -> list[dict]:
        results: dict[int, dict] = {}
        for page in range(1, max(1, pages) + 1):
            try:
                payload = self.get(
                    f"/movie/{int(tmdb_id)}/recommendations",
                    page=page,
                )
            except TmdbError as exc:
                # Certaines anciennes associations IMDb/TMDB pointent vers une
                # série ou une fiche supprimée. Ce n'est pas une panne du
                # moteur de recommandation : cette graine est simplement
                # inutilisable.
                if exc.status_code == 404:
                    break
                raise
            for item in payload.get("results", []):
                results[int(item["id"])] = item
            if page >= int(payload.get("total_pages", page)):
                break
        return list(results.values())

    def movie_similar(
        self,
        tmdb_id: int,
        *,
        pages: int = 1,
    ) -> list[dict]:
        results: dict[int, dict] = {}
        for page in range(1, max(1, pages) + 1):
            try:
                payload = self.get(
                    f"/movie/{int(tmdb_id)}/similar",
                    page=page,
                )
            except TmdbError as exc:
                if exc.status_code == 404:
                    break
                raise
            for item in payload.get("results", []):
                results[int(item["id"])] = item
            if page >= int(payload.get("total_pages", page)):
                break
        return list(results.values())


def _upsert_person(
    connection,
    person: dict,
    imdb_id: str,
    role: str,
    job: str = "",
    character_name: str = "",
    credit_order: int | None = None,
) -> None:
    person_id = person.get("id")
    name = person.get("name")
    if person_id is None or not name:
        return
    connection.execute(
        """
        INSERT INTO people(tmdb_id, name, known_for_department)
        VALUES (?, ?, ?)
        ON CONFLICT(tmdb_id) DO UPDATE SET
          name=excluded.name,
          known_for_department=COALESCE(
            excluded.known_for_department, people.known_for_department
          )
        """,
        (person_id, name, person.get("known_for_department")),
    )
    connection.execute(
        """
        INSERT OR REPLACE INTO credits(
          imdb_id, person_id, role, job, character_name, credit_order
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            imdb_id,
            person_id,
            role,
            job or "",
            character_name or "",
            credit_order,
        ),
    )


def _resolve_named_entity_id(
    connection,
    table: str,
    tmdb_id: int,
    name: str,
) -> int:
    """Insert a TMDB named entity and safely resolve rare name collisions."""

    if table not in {"genres", "keywords"}:
        raise ValueError(f"Table d’entités non prise en charge : {table}")
    existing_id = connection.execute(
        f"SELECT tmdb_id FROM {table} WHERE tmdb_id=?",
        (int(tmdb_id),),
    ).fetchone()
    if existing_id:
        return int(existing_id["tmdb_id"])
    existing_name = connection.execute(
        f"SELECT tmdb_id FROM {table} WHERE name=?",
        (str(name),),
    ).fetchone()
    if existing_name:
        return int(existing_name["tmdb_id"])
    connection.execute(
        f"INSERT INTO {table}(tmdb_id, name) VALUES (?, ?)",
        (int(tmdb_id), str(name)),
    )
    return int(tmdb_id)


def _store_details(
    database: str | Path | None,
    imdb_id: str,
    media_type: str,
    details: dict,
    region: str,
) -> None:
    title = details.get("title") or details.get("name")
    original_title = details.get("original_title") or details.get("original_name")
    release_date = details.get("release_date") or details.get("first_air_date")
    runtime = details.get("runtime")
    if runtime is None and details.get("episode_run_time"):
        runtime = details["episode_run_time"][0]

    with transaction(database) as connection:
        connection.execute(
            """
            UPDATE titles SET
              tmdb_id=?,
              title=COALESCE(?, title),
              original_title=COALESCE(?, original_title),
              title_type=COALESCE(title_type, ?),
              runtime_minutes=COALESCE(?, runtime_minutes),
              release_date=COALESCE(?, release_date),
              overview=?,
              tagline=?,
              original_language=?,
              countries_json=?,
              companies_json=?,
              poster_path=?,
              tmdb_rating=?,
              tmdb_vote_count=?,
              popularity=?,
              metadata_status='done',
              metadata_error=NULL,
              enriched_at=?
            WHERE imdb_id=?
            """,
            (
                details.get("id"),
                title,
                original_title,
                media_type,
                runtime,
                release_date,
                details.get("overview"),
                details.get("tagline"),
                details.get("original_language"),
                json.dumps(details.get("production_countries", []), ensure_ascii=False),
                json.dumps(details.get("production_companies", []), ensure_ascii=False),
                details.get("poster_path"),
                details.get("vote_average"),
                details.get("vote_count"),
                details.get("popularity"),
                datetime.now(UTC).isoformat(),
                imdb_id,
            ),
        )

        connection.execute("DELETE FROM title_genres WHERE imdb_id=?", (imdb_id,))
        for genre in details.get("genres", []):
            genre_id = _resolve_named_entity_id(
                connection,
                "genres",
                int(genre["id"]),
                str(genre["name"]),
            )
            connection.execute(
                "INSERT OR IGNORE INTO title_genres(imdb_id, genre_id) VALUES (?, ?)",
                (imdb_id, genre_id),
            )

        connection.execute("DELETE FROM credits WHERE imdb_id=?", (imdb_id,))
        credits = details.get("credits", {})
        # Les vingt premiers interprètes suffisent pour le modèle et évitent
        # qu'un figurant pèse comme un rôle principal.
        for actor in credits.get("cast", [])[:20]:
            _upsert_person(
                connection,
                actor,
                imdb_id,
                role="cast",
                job="Actor",
                character_name=actor.get("character", ""),
                credit_order=actor.get("order"),
            )
        relevant_jobs = {
            "Director",
            "Screenplay",
            "Writer",
            "Story",
            "Director of Photography",
            "Original Music Composer",
            "Editor",
        }
        for crew in credits.get("crew", []):
            job = crew.get("job", "")
            if job not in relevant_jobs:
                continue
            role = {
                "Director": "director",
                "Screenplay": "writer",
                "Writer": "writer",
                "Story": "writer",
                "Director of Photography": "cinematography",
                "Original Music Composer": "composer",
                "Editor": "editor",
            }[job]
            _upsert_person(connection, crew, imdb_id, role=role, job=job)

        connection.execute("DELETE FROM title_keywords WHERE imdb_id=?", (imdb_id,))
        keywords_payload = details.get("keywords", {})
        keyword_items = keywords_payload.get("keywords", keywords_payload.get("results", []))
        for keyword in keyword_items:
            keyword_id = _resolve_named_entity_id(
                connection,
                "keywords",
                int(keyword["id"]),
                str(keyword["name"]),
            )
            connection.execute(
                "INSERT OR IGNORE INTO title_keywords(imdb_id, keyword_id) VALUES (?, ?)",
                (imdb_id, keyword_id),
            )

        connection.execute(
            "DELETE FROM providers WHERE imdb_id=? AND region=?",
            (imdb_id, region),
        )
        region_payload = (
            details.get("watch/providers", {}).get("results", {}).get(region, {})
        )
        for access_type in ("flatrate", "free", "ads", "rent", "buy"):
            for provider in region_payload.get(access_type, []):
                connection.execute(
                    """
                    INSERT OR REPLACE INTO providers(
                      imdb_id, region, provider_id, provider_name, access_type
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        imdb_id,
                        region,
                        provider["provider_id"],
                        provider["provider_name"],
                        access_type,
                    ),
                )


def enrich_library(
    client: TmdbClient,
    database: str | Path | None = None,
    *,
    limit: int | None = None,
    retry_errors: bool = True,
    on_progress: Callable[[EnrichmentProgress], None] | None = None,
) -> dict[str, int]:
    initialize(database)
    statuses = ("pending", "error") if retry_errors else ("pending",)
    placeholders = ",".join("?" for _ in statuses)
    sql = (
        "SELECT imdb_id, title FROM titles "
        f"WHERE metadata_status IN ({placeholders}) "
        "AND imdb_id LIKE 'tt%' ORDER BY date_rated DESC, imdb_id"
    )
    params: list[object] = list(statuses)
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
    with connect(database) as connection:
        rows = connection.execute(sql, params).fetchall()

    counts = {"done": 0, "not_found": 0, "error": 0}
    total = len(rows)
    for index, row in enumerate(rows, start=1):
        imdb_id = row["imdb_id"]
        title = row["title"]
        status = "error"
        try:
            resolved = client.resolve_imdb_id(imdb_id)
            if resolved is None:
                with transaction(database) as connection:
                    connection.execute(
                        """
                        UPDATE titles
                        SET metadata_status='not_found', metadata_error=?
                        WHERE imdb_id=?
                        """,
                        ("Identifiant IMDb absent de TMDB", imdb_id),
                    )
                status = "not_found"
            else:
                media_type, tmdb_id = resolved
                details = client.details(media_type, tmdb_id)
                _store_details(database, imdb_id, media_type, details, client.region)
                status = "done"
        except TmdbError as exc:
            # Une erreur globale d’authentification, de quota ou de réseau ne
            # doit pas transformer silencieusement tout l’historique en erreurs
            # individuelles. Seule une fiche réellement absente est locale.
            if exc.status_code != 404:
                raise
            with transaction(database) as connection:
                connection.execute(
                    """
                    UPDATE titles
                    SET metadata_status='not_found', metadata_error=?
                    WHERE imdb_id=?
                    """,
                    (str(exc)[:1000], imdb_id),
                )
            status = "not_found"
        except Exception as exc:  # donnée isolée invalide, reprise possible
            with transaction(database) as connection:
                connection.execute(
                    """
                    UPDATE titles
                    SET metadata_status='error', metadata_error=?
                    WHERE imdb_id=?
                    """,
                    (str(exc)[:1000], imdb_id),
                )
            status = "error"
        counts[status] += 1
        if on_progress:
            on_progress(
                EnrichmentProgress(
                    processed=index,
                    total=total,
                    imdb_id=imdb_id,
                    title=title,
                    status=status,
                )
            )
    return counts


def refresh_library_metadata(
    client: TmdbClient,
    database: str | Path | None = None,
    *,
    limit: int | None = None,
    on_progress: Callable[[EnrichmentProgress], None] | None = None,
) -> dict[str, int]:
    """Rafraîchit les fiches déjà reliées à TMDB, notamment leur traduction."""
    initialize(database)
    sql = """
        SELECT imdb_id, title, title_type, tmdb_id
        FROM titles
        WHERE metadata_status='done' AND tmdb_id IS NOT NULL
        ORDER BY enriched_at ASC, imdb_id
    """
    params: list[object] = []
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
    with connect(database) as connection:
        rows = connection.execute(sql, params).fetchall()

    counts = {"done": 0, "error": 0}
    total = len(rows)
    for index, row in enumerate(rows, start=1):
        status = "error"
        try:
            media_type = tmdb_media_type(row["title_type"])
            details = client.details(media_type, int(row["tmdb_id"]))
            _store_details(
                database,
                row["imdb_id"],
                media_type,
                details,
                client.region,
            )
            status = "done"
        except TmdbError as exc:
            if exc.status_code != 404:
                raise
            with transaction(database) as connection:
                connection.execute(
                    """
                    UPDATE titles SET metadata_error=?
                    WHERE imdb_id=?
                    """,
                    (str(exc)[:1000], row["imdb_id"]),
                )
        except Exception as exc:
            with transaction(database) as connection:
                connection.execute(
                    """
                    UPDATE titles SET metadata_error=?
                    WHERE imdb_id=?
                    """,
                    (str(exc)[:1000], row["imdb_id"]),
                )
        counts[status] += 1
        if on_progress:
            on_progress(
                EnrichmentProgress(
                    processed=index,
                    total=total,
                    imdb_id=row["imdb_id"],
                    title=row["title"],
                    status=status,
                )
            )
    return counts


def enrich_candidates(client: TmdbClient, candidates: Iterable[dict]) -> list[dict]:
    enriched: list[dict] = []
    for candidate in candidates:
        try:
            details = client.details("movie", int(candidate["id"]))
        except TmdbError as exc:
            if exc.status_code == 404:
                continue
            raise
        details["_discover"] = candidate
        enriched.append(details)
    return enriched

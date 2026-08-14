from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Callable

from .db import connect
from .media_types import is_series_type
from .public_rating import best_public_rating
from .tmdb import TmdbError


SOURCE_POPULARITY = "Sorties et popularité"
SOURCE_QUALITY = "Films publics solides"
SOURCE_FAVORITES = "Films que tu as aimés"
SOURCE_SIMILAR = "Films similaires à tes favoris"
SOURCE_SEMANTIC = "Histoires proches de tes goûts"
SOURCE_CREATORS = "Réalisateurs et scénaristes appréciés"
SOURCE_ACTORS = "Acteurs principaux appréciés"
SOURCE_KEYWORDS = "Thèmes appréciés"
SOURCE_GENRES = "Genres appréciés"

SOURCE_PRIORITY = (
    SOURCE_FAVORITES,
    SOURCE_SIMILAR,
    SOURCE_SEMANTIC,
    SOURCE_CREATORS,
    SOURCE_KEYWORDS,
    SOURCE_QUALITY,
    SOURCE_POPULARITY,
    SOURCE_GENRES,
    SOURCE_ACTORS,
)


def vote_threshold(release_date: str | None, reliability: str) -> int:
    if reliability == "Souple":
        return 0
    try:
        released = date.fromisoformat(release_date or "")
        age_days = max(0, (date.today() - released).days)
    except ValueError:
        age_days = 3650
    if reliability == "Forte":
        if age_days < 180:
            return 25
        if age_days < 730:
            return 120
        if age_days < 1825:
            return 400
        return 1200
    if age_days < 180:
        return 8
    if age_days < 730:
        return 30
    if age_days < 1825:
        return 100
    return 250


def passes_vote_filter(candidate: dict, reliability: str) -> bool:
    return int(candidate.get("vote_count") or 0) >= vote_threshold(
        candidate.get("release_date"),
        reliability,
    )


def passes_date_filter(
    candidate: dict,
    start_date: str | None,
    end_date: str | None,
) -> bool:
    if not start_date and not end_date:
        return True
    try:
        released = date.fromisoformat(candidate.get("release_date") or "")
    except ValueError:
        return False
    if start_date and released < date.fromisoformat(start_date):
        return False
    if end_date and released > date.fromisoformat(end_date):
        return False
    return True


def merge_candidate(
    pool: dict[int, dict],
    candidate: dict,
    source: str,
) -> None:
    candidate_id = int(candidate["id"])
    if candidate_id not in pool:
        pool[candidate_id] = {**candidate, "_sources": [source]}
    elif source not in pool[candidate_id]["_sources"]:
        pool[candidate_id]["_sources"].append(source)


def _positive_rows(
    profile: dict,
    dimension: str,
    *,
    minimum_seen: int,
) -> list[dict]:
    return sorted(
        [
            row
            for row in profile.get("dimensions", {}).get(dimension, [])
            if float(row.get("affinity", 0.0)) > 0
            and int(row.get("seen", 0)) >= minimum_seen
            and str(row.get("id", "")).isdigit()
        ],
        key=lambda row: (
            float(row.get("affinity", 0.0)),
            float(row.get("confidence", 0.0)),
            int(row.get("seen", 0)),
        ),
        reverse=True,
    )


def profile_people(
    profile: dict,
    *,
    creators_limit: int,
    actors_limit: int,
) -> tuple[list[int], list[int]]:
    creators: list[int] = []
    for dimension in ("directors", "writers"):
        for row in _positive_rows(profile, dimension, minimum_seen=2):
            person_id = int(row["id"])
            if person_id not in creators:
                creators.append(person_id)
    creator_rows = {
        int(row["id"]): row
        for dimension in ("directors", "writers")
        for row in _positive_rows(profile, dimension, minimum_seen=2)
    }
    creators.sort(
        key=lambda person_id: (
            float(creator_rows[person_id].get("affinity", 0.0)),
            int(creator_rows[person_id].get("seen", 0)),
        ),
        reverse=True,
    )
    actors = [
        int(row["id"])
        for row in _positive_rows(profile, "actors", minimum_seen=5)
    ]
    return creators[:creators_limit], actors[:actors_limit]


def profile_dimension_ids(
    profile: dict,
    dimension: str,
    limit: int,
    *,
    minimum_seen: int = 2,
) -> list[int]:
    return [
        int(row["id"])
        for row in _positive_rows(
            profile,
            dimension,
            minimum_seen=minimum_seen,
        )[:limit]
    ]


def favorite_seeds(
    database: str | Path | None,
    limit: int,
) -> list[int]:
    """Select unusually loved films, then diversify their creative signature."""

    with connect(database) as connection:
        rows = connection.execute(
            """
            SELECT
              t.tmdb_id, t.title_type, t.user_rating,
              t.imdb_rating, t.num_votes, t.tmdb_rating,
              t.tmdb_vote_count, t.date_rated,
              GROUP_CONCAT(DISTINCT tg.genre_id) AS genre_ids,
              GROUP_CONCAT(DISTINCT CASE
                WHEN c.role IN ('director', 'writer') THEN c.person_id
              END) AS creator_ids
            FROM titles AS t
            LEFT JOIN title_genres AS tg ON tg.imdb_id=t.imdb_id
            LEFT JOIN credits AS c ON c.imdb_id=t.imdb_id
            WHERE t.tmdb_id IS NOT NULL AND t.user_rating >= 8
            GROUP BY t.imdb_id
            """
        ).fetchall()
    candidates: list[dict] = []
    for row in rows:
        if is_series_type(row["title_type"]):
            continue
        evidence = best_public_rating(
            imdb_rating=row["imdb_rating"],
            imdb_votes=row["num_votes"],
            tmdb_rating=row["tmdb_rating"],
            tmdb_votes=row["tmdb_vote_count"],
        )
        public = (
            float(evidence.adjusted_rating)
            if evidence.adjusted_rating is not None
            else 6.8
        )
        candidates.append(
            {
                "tmdb_id": int(row["tmdb_id"]),
                "residual": float(row["user_rating"]) - public,
                "rating": float(row["user_rating"]),
                "date_rated": str(row["date_rated"] or ""),
                "genres": {
                    value
                    for value in str(row["genre_ids"] or "").split(",")
                    if value
                },
                "creators": {
                    value
                    for value in str(row["creator_ids"] or "").split(",")
                    if value
                },
            }
        )
    candidates.sort(
        key=lambda row: (
            float(row["residual"]),
            float(row["rating"]),
            str(row["date_rated"]),
        ),
        reverse=True,
    )
    chosen: list[int] = []
    used_genres: set[str] = set()
    used_creators: set[str] = set()
    remaining = list(candidates)
    # First pass: maximize preference signal while avoiding fifteen near-
    # identical seeds from one prolific franchise, genre or director.
    while remaining and len(chosen) < limit:
        best = max(
            remaining,
            key=lambda row: (
                float(row["residual"])
                - 0.30 * len(row["genres"] & used_genres)
                - 0.45 * len(row["creators"] & used_creators),
                float(row["rating"]),
            ),
        )
        remaining.remove(best)
        chosen.append(int(best["tmdb_id"]))
        used_genres.update(best["genres"])
        used_creators.update(best["creators"])
    return chosen


def _candidate_priority(candidate: dict) -> tuple[float, float, float]:
    return (
        float(candidate.get("_retrieval_utility") or 0.0),
        float(len(candidate.get("_sources", []))),
        float(candidate.get("popularity") or 0.0),
    )


def personalize_candidate_order(
    candidates: list[dict],
    evidence: dict[int, dict],
    *,
    maximum_semantic_source: int = 120,
) -> tuple[list[dict], int]:
    """Place locally similar stories early without discarding alternatives.

    Retrieval is deliberately independent from the final predicted score.  It
    only decides which candidates deserve a complete TMDB enrichment when the
    raw pool is larger than the analysis budget.
    """

    ranked: list[tuple[float, dict]] = []
    for candidate in candidates:
        row = evidence.get(int(candidate["id"]), {})
        score = row.get("score")
        confidence = float(row.get("confidence") or 0.0)
        utility = (
            0.85 * float(score) + 0.15 * confidence
            if score is not None
            else 0.0
        )
        candidate["_retrieval_score"] = (
            round(100.0 * float(score), 1) if score is not None else None
        )
        candidate["_retrieval_confidence"] = round(100.0 * confidence, 1)
        candidate["_retrieval_utility"] = utility
        ranked.append((utility, candidate))

    usable = sorted(
        [row for row in ranked if row[0] > 0.0],
        key=lambda row: row[0],
        reverse=True,
    )
    semantic_count = min(
        maximum_semantic_source,
        max(30, len(candidates) // 3),
        len(usable),
    )
    for _, candidate in usable[:semantic_count]:
        if SOURCE_SEMANTIC not in candidate.get("_sources", []):
            candidate.setdefault("_sources", []).append(SOURCE_SEMANTIC)
    return balanced_candidate_order(candidates), semantic_count


def balanced_candidate_order(candidates: list[dict]) -> list[dict]:
    """Round-robin sources so popularity cannot consume the entire budget."""

    buckets: dict[str, list[dict]] = {}
    for source in SOURCE_PRIORITY:
        bucket = [
            candidate
            for candidate in candidates
            if source in candidate.get("_sources", [])
        ]
        bucket.sort(key=_candidate_priority, reverse=True)
        buckets[source] = bucket
    selected: list[dict] = []
    selected_ids: set[int] = set()
    while len(selected) < len(candidates):
        added = False
        for source in SOURCE_PRIORITY:
            bucket = buckets[source]
            while bucket and int(bucket[0]["id"]) in selected_ids:
                bucket.pop(0)
            if not bucket:
                continue
            candidate = bucket.pop(0)
            candidate_id = int(candidate["id"])
            selected.append(candidate)
            selected_ids.add(candidate_id)
            added = True
        if not added:
            break
    leftovers = [
        candidate
        for candidate in sorted(
            candidates,
            key=_candidate_priority,
            reverse=True,
        )
        if int(candidate["id"]) not in selected_ids
    ]
    return selected + leftovers


def selected_source_counts(candidates: list[dict], limit: int) -> dict[str, int]:
    counts = {source: 0 for source in SOURCE_PRIORITY}
    for candidate in candidates[:limit]:
        for source in candidate.get("_sources", []):
            counts[source] = counts.get(source, 0) + 1
    return {source: count for source, count in counts.items() if count}


def build_candidate_pool(
    client,
    profile: dict,
    database: str | Path | None,
    *,
    start_date: str | None,
    end_date: str | None,
    settings: dict,
    reliability: str,
    excluded_genre_ids: set[int] | None,
    excluded_genre: Callable[[dict, set[int] | None], bool],
    trace_ids: set[int] | None = None,
) -> tuple[list[dict], dict[str, int], dict[str, object]]:
    pool: dict[int, dict] = {}
    source_counts: dict[str, int] = {}

    def add(source: str, items: list[dict]) -> None:
        source_counts[source] = source_counts.get(source, 0) + len(items)
        for candidate in items:
            merge_candidate(pool, candidate, source)

    add(
        SOURCE_POPULARITY,
        client.discover_recent_movies(
            start_date,
            end_date,
            pages=settings["discover_pages"],
            min_votes=0,
        ),
    )
    if settings.get("quality_pages", 0):
        add(
            SOURCE_QUALITY,
            client.discover_recent_movies(
                start_date,
                end_date,
                pages=settings["quality_pages"],
                min_votes=vote_threshold(end_date, reliability),
                sort_by="vote_average.desc",
            ),
        )

    seed_errors = 0
    seeds = favorite_seeds(database, settings["seed_count"])
    for seed in seeds:
        try:
            add(
                SOURCE_FAVORITES,
                client.movie_recommendations(
                    seed,
                    pages=int(settings.get("recommendation_pages", 1)),
                ),
            )
            if hasattr(client, "movie_similar"):
                add(
                    SOURCE_SIMILAR,
                    client.movie_similar(
                        seed,
                        pages=int(settings.get("similar_pages", 1)),
                    ),
                )
        except TmdbError as exc:
            if exc.status_code == 404:
                seed_errors += 1
                continue
            raise
    if seed_errors:
        source_counts["Graines TMDB ignorées"] = seed_errors

    creators, actors = profile_people(
        profile,
        creators_limit=settings["creator_count"],
        actors_limit=settings["actor_count"],
    )
    for person_id in creators:
        add(
            SOURCE_CREATORS,
            client.discover_recent_movies(
                start_date,
                end_date,
                pages=1,
                min_votes=0,
                with_people=person_id,
            ),
        )
    for person_id in actors:
        add(
            SOURCE_ACTORS,
            client.discover_recent_movies(
                start_date,
                end_date,
                pages=1,
                min_votes=0,
                with_people=person_id,
            ),
        )
    for keyword_id in profile_dimension_ids(
        profile,
        "keywords",
        settings["keyword_count"],
    ):
        add(
            SOURCE_KEYWORDS,
            client.discover_recent_movies(
                start_date,
                end_date,
                pages=1,
                min_votes=0,
                with_keywords=keyword_id,
            ),
        )
    for genre_id in profile_dimension_ids(
        profile,
        "genres",
        settings["genre_count"],
    ):
        add(
            SOURCE_GENRES,
            client.discover_recent_movies(
                start_date,
                end_date,
                pages=1,
                min_votes=0,
                with_genres=genre_id,
            ),
        )

    all_candidates = list(pool.values())
    in_window = [
        candidate
        for candidate in all_candidates
        if passes_date_filter(candidate, start_date, end_date)
    ]
    reliable = [
        candidate
        for candidate in in_window
        if passes_vote_filter(candidate, reliability)
    ]
    filtered = [
        candidate
        for candidate in reliable
        if not excluded_genre(candidate, excluded_genre_ids)
    ]
    ordered = balanced_candidate_order(filtered)
    diagnostics: dict[str, object] = {
        "raw_unique_candidates": len(all_candidates),
        "excluded_outside_window": len(all_candidates) - len(in_window),
        "excluded_insufficient_votes": len(in_window) - len(reliable),
        "excluded_genres": len(reliable) - len(filtered),
        "after_pre_enrichment_filters": len(filtered),
    }
    if trace_ids:
        in_window_ids = {int(candidate["id"]) for candidate in in_window}
        reliable_ids = {int(candidate["id"]) for candidate in reliable}
        filtered_ids = {int(candidate["id"]) for candidate in filtered}
        ranks = {
            int(candidate["id"]): position
            for position, candidate in enumerate(ordered, start=1)
        }
        trace: dict[str, dict] = {}
        for candidate_id in sorted({int(value) for value in trace_ids}):
            candidate = pool.get(candidate_id)
            if candidate is None:
                state = "absent_from_all_sources"
            elif candidate_id not in in_window_ids:
                state = "outside_release_window"
            elif candidate_id not in reliable_ids:
                state = "insufficient_votes"
            elif candidate_id not in filtered_ids:
                state = "excluded_genre"
            else:
                state = "eligible"
            trace[str(candidate_id)] = {
                "state": state,
                "rank": ranks.get(candidate_id),
                "sources": (
                    list(candidate.get("_sources", [])) if candidate else []
                ),
                "release_date": (
                    candidate.get("release_date") if candidate else None
                ),
                "vote_count": (
                    int(candidate.get("vote_count") or 0) if candidate else None
                ),
            }
        diagnostics["candidate_trace"] = trace
    return ordered, source_counts, diagnostics

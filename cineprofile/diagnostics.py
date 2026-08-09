from __future__ import annotations

import json
import logging
from copy import deepcopy
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any


DIAGNOSTIC_SCHEMA_VERSION = 4
LOGGER_NAME = "cineprofile"


def _database_path(database: str | Path | None) -> Path:
    return Path(database or "data/cineprofile.db")


def _logs_directory(database: str | Path | None) -> Path:
    directory = _database_path(database).parent / "logs"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def configure_logging(database: str | Path | None) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    target = (_logs_directory(database) / "cineprofile.log").resolve()
    for handler in list(logger.handlers):
        if (
            isinstance(handler, RotatingFileHandler)
            and Path(handler.baseFilename).resolve() == target
        ):
            return logger
        if isinstance(handler, RotatingFileHandler):
            logger.removeHandler(handler)
            handler.close()
    handler = RotatingFileHandler(
        target,
        maxBytes=2_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def new_search_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")


def _candidate_diagnostic(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "rank": item.get("diagnostic_rank"),
        "tmdb_id": item.get("tmdb_id"),
        "imdb_id": item.get("imdb_id"),
        "title": item.get("title"),
        "release_date": item.get("release_date"),
        "genres": item.get("genres", []),
        "overview_available": bool(item.get("overview")),
        "overview_length": len(str(item.get("overview") or "")),
        "tmdb_rating_raw": item.get("vote_average"),
        "tmdb_vote_count": item.get("vote_count"),
        "public_rating_adjusted": item.get("public_rating_adjusted"),
        "public_rating_reliability": item.get("public_rating_reliability"),
        "raw_like_probability": item.get("raw_like_probability"),
        "like_probability": item.get("like_probability"),
        "base_like_rate_percent": item.get("base_like_rate_percent"),
        "like_probability_lift_points": item.get(
            "like_probability_lift_points"
        ),
        "like_probability_lift_ratio": item.get(
            "like_probability_lift_ratio"
        ),
        "predicted_rating": item.get("predicted_rating"),
        "prediction_interval": [
            item.get("prediction_low"),
            item.get("prediction_high"),
        ],
        "confidence": item.get("confidence"),
        "model_coverage": item.get("model_coverage"),
        "personal_adjustment": item.get("personal_adjustment"),
        "user_baseline_rating": item.get("user_baseline_rating"),
        "personal_engine": item.get("personal_engine"),
        "personal_variant_label": item.get("personal_variant_label"),
        "underlying_personal_engine": item.get("underlying_personal_engine"),
        "local_like_probability": item.get("local_like_probability"),
        "global_like_probability": item.get("global_like_probability"),
        "local_neighbor_weight": item.get("local_neighbor_weight"),
        "public_influence_weight": item.get("public_influence_weight"),
        "match_tier": item.get("match_tier"),
        "interest_model_version": item.get("interest_model_version"),
        "interest_score": item.get("interest_score"),
        "interest_label": item.get("interest_label"),
        "interest_confidence": item.get("interest_confidence"),
        "interest_positive_reasons": item.get(
            "interest_positive_reasons", []
        ),
        "interest_reservations": item.get("interest_reservations", []),
        "satisfaction_lift_index": item.get("satisfaction_lift_index"),
        "recommendation_score": item.get("recommendation_score"),
        "affinity_index": item.get("affinity_index"),
        "safe_score": item.get("safe_score"),
        "safe_value_score": item.get("safe_value_score"),
        "safe_eligibility_tier": item.get("safe_eligibility_tier"),
        "safe_eligibility_label": item.get("safe_eligibility_label"),
        "rank_score": item.get("rank_score"),
        "discovery_score": item.get("discovery_score"),
        "recommended_rank": item.get("recommended_rank"),
        "ranking_mode": item.get("ranking_mode"),
        "collection_id": item.get("collection_id"),
        "semantic_engine": item.get("semantic_engine"),
        "semantic_neighbors": item.get("semantic_neighbors", [])[:5],
        "retrieval_score": item.get("retrieval_score"),
        "retrieval_confidence": item.get("retrieval_confidence"),
        "negative_genres": item.get("negative_genres", []),
        "positive_signals": item.get("learned_positive_signals", [])[:8],
        "negative_signals": item.get("learned_negative_signals", [])[:8],
        "reasons": item.get("reasons", []),
        "sources": item.get("sources", []),
    }


def _automated_checks(ranked: list[dict[str, Any]]) -> list[dict[str, Any]]:
    top_ten = ranked[:10]
    low_confidence_high_probability = [
        item["title"]
        for item in top_ten
        if float(item.get("like_probability") or 0) >= 80
        and float(item.get("confidence") or 0) < 45
    ]
    missing_overviews = [
        item["title"] for item in top_ten if not item["overview_available"]
    ]
    low_vote_titles = [
        item["title"]
        for item in top_ten
        if int(item.get("tmdb_vote_count") or 0) < 100
    ]
    engines = {
        str(item.get("personal_engine"))
        for item in ranked
        if item.get("personal_engine")
    }
    repeated_collections = [
        collection_id
        for collection_id in {
            item.get("collection_id")
            for item in top_ten
            if item.get("collection_id") is not None
        }
        if sum(
            item.get("collection_id") == collection_id
            for item in top_ten
        )
        >= 3
    ]
    recommendation_gap_violations: list[str] = []
    for index, item in enumerate(top_ten):
        later = ranked[index + 1 :]
        if not later:
            continue
        strongest_later = max(
            float(
                row.get(
                    "recommendation_score",
                    row.get("like_probability") or 0.0,
                )
            )
            for row in later
        )
        if (
            float(
                item.get(
                    "recommendation_score",
                    item.get("like_probability") or 0.0,
                )
            )
            < strongest_later - 8.0
            and item.get("ranking_mode") != "Valeurs sûres"
        ):
            recommendation_gap_violations.append(
                str(item.get("title") or "")
            )
    popularity_only = [
        item["title"]
        for item in top_ten
        if item.get("sources") == ["Sorties et popularité"]
    ]
    interest_reservation_counts: dict[str, int] = {}
    for item in top_ten:
        for reservation in item.get("interest_reservations", []):
            factor = str(reservation.get("factor") or "autre")
            interest_reservation_counts[factor] = (
                interest_reservation_counts.get(factor, 0) + 1
            )
    saturated_reservations = {
        factor: count
        for factor, count in interest_reservation_counts.items()
        if count >= 5
    }
    direct_public_influence = [
        item["title"]
        for item in top_ten
        if float(item.get("public_influence_weight") or 0.0) > 0.15
    ]
    strong_matches = [
        item["title"]
        for item in ranked
        if item.get("match_tier") == "Correspondance solide"
    ]
    high_interest = [
        item["title"]
        for item in ranked
        if float(item.get("interest_score") or 0.0) >= 70.0
    ]
    has_interest = any(
        item.get("interest_score") is not None for item in ranked
    )
    satisfaction_without_interest = [
        item["title"]
        for item in top_ten
        if float(item.get("like_probability") or 0.0) >= 30.0
        and float(item.get("interest_score") or 0.0) < 45.0
    ]
    return [
        {
            "name": "un seul moteur personnel actif",
            "status": "pass" if len(engines) <= 1 else "warning",
            "engines": sorted(engines),
        },
        {
            "name": "probabilités élevées avec faible confiance",
            "status": (
                "pass" if not low_confidence_high_probability else "warning"
            ),
            "titles": low_confidence_high_probability,
        },
        {
            "name": "résumés absents dans le top 10",
            "status": "pass" if not missing_overviews else "warning",
            "titles": missing_overviews,
        },
        {
            "name": "moins de 100 votes TMDB dans le top 10",
            "status": "pass" if not low_vote_titles else "warning",
            "titles": low_vote_titles,
        },
        {
            "name": "franchise répétée au moins trois fois dans le top 10",
            "status": "pass" if not repeated_collections else "warning",
            "collection_ids": repeated_collections,
        },
        {
            "name": "promotion au-delà de la marge combinée de 8 points",
            "status": (
                "pass" if not recommendation_gap_violations else "warning"
            ),
            "titles": recommendation_gap_violations,
        },
        {
            "name": "popularité seule dominante dans le top 10",
            "status": "pass" if len(popularity_only) < 7 else "warning",
            "count": len(popularity_only),
            "titles": popularity_only,
        },
        {
            "name": "même frein d’envie présent au moins cinq fois dans le top 10",
            "status": (
                "pass" if not saturated_reservations else "warning"
            ),
            "factors": saturated_reservations,
        },
        {
            "name": "influence publique directe supérieure à 15 %",
            "status": "pass" if not direct_public_influence else "warning",
            "titles": direct_public_influence,
        },
        {
            "name": "au moins une correspondance solide",
            "status": "pass" if strong_matches else "warning",
            "count": len(strong_matches),
        },
        {
            "name": "au moins un film avec une envie élevée",
            "status": (
                "pass" if high_interest or not has_interest else "warning"
            ),
            "count": len(high_interest),
            "titles": high_interest[:10],
        },
        {
            "name": "chance d’un 8+ élevée mais envie faible dans le top 10",
            "status": (
                "pass" if not satisfaction_without_interest else "warning"
            ),
            "titles": satisfaction_without_interest,
        },
    ]


def build_search_diagnostic(
    *,
    app_version: str,
    search_id: str,
    profile: dict[str, Any],
    diagnostics: dict[str, Any],
    results: list[dict[str, Any]],
    settings: dict[str, Any],
) -> dict[str, Any]:
    ranked = []
    for rank, item in enumerate(results, start=1):
        copy = dict(item)
        copy["diagnostic_rank"] = rank
        ranked.append(_candidate_diagnostic(copy))
    return {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "search_id": search_id,
        "app_version": app_version,
        "profile_model_version": profile.get("model_version"),
        "personal_model": profile.get("personal_model", {}),
        "settings": settings,
        "pipeline": diagnostics,
        "automated_checks": _automated_checks(ranked),
        "recommendations": ranked,
        "privacy": (
            "Le jeton TMDB, le chemin complet de la base et le contenu intégral "
            "de l’historique ne sont pas inclus."
        ),
    }


def diagnostic_with_ui_view(
    payload: dict[str, Any],
    ui_view: dict[str, Any],
    *,
    view_recommendations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return a diagnostic whose checks match the order visible in the UI."""

    result = deepcopy(payload)
    result.setdefault(
        "pipeline_automated_checks",
        deepcopy(result.get("automated_checks", [])),
    )
    result["ui_view"] = deepcopy(ui_view)
    if view_recommendations is not None:
        view_by_tmdb_id = {
            int(item["tmdb_id"]): item
            for item in view_recommendations
            if item.get("tmdb_id") is not None
        }
        for item in result.get("recommendations", []):
            tmdb_id = item.get("tmdb_id")
            if tmdb_id is None:
                continue
            view_item = view_by_tmdb_id.get(int(tmdb_id))
            if view_item is None:
                continue
            for key in (
                "recommended_rank",
                "ranking_mode",
                "recommendation_view",
                "safe_value_score",
                "safe_eligibility_tier",
                "safe_eligibility_label",
                "discovery_rank",
                "safe_rank",
            ):
                if key in view_item:
                    item[key] = deepcopy(view_item[key])
    by_tmdb_id = {
        int(item["tmdb_id"]): item
        for item in result.get("recommendations", [])
        if item.get("tmdb_id") is not None
    }
    ordered = [
        by_tmdb_id[int(tmdb_id)]
        for tmdb_id in ui_view.get("displayed_tmdb_ids", [])
        if int(tmdb_id) in by_tmdb_id
    ]
    result["automated_checks"] = _automated_checks(ordered)
    result["automated_checks_scope"] = "résultats actuellement affichés"
    return result


def save_search_diagnostic(
    database: str | Path | None,
    payload: dict[str, Any],
) -> Path:
    search_id = str(payload["search_id"])
    path = _logs_directory(database) / f"diagnostic_{search_id}.json"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)
    logger = configure_logging(database)
    logger.info(
        "search_completed | search_id=%s | candidates=%s | returned=%s | "
        "engine=%s | diagnostic=%s",
        search_id,
        payload.get("pipeline", {}).get("unique_candidates"),
        len(payload.get("recommendations", [])),
        payload.get("pipeline", {}).get("personal_engine"),
        path.name,
    )
    for item in payload.get("recommendations", [])[:30]:
        logger.info(
            "candidate | search_id=%s | rank=%s | tmdb_id=%s | title=%s | "
            "raw_rating=%s | votes=%s | adjusted_rating=%s | raw_like=%s | "
            "like=%s | interest=%s | recommendation=%s | confidence=%s | "
            "coverage=%s | engine=%s | safe_score=%s | rank_score=%s | "
            "sources=%s",
            search_id,
            item.get("rank"),
            item.get("tmdb_id"),
            item.get("title"),
            item.get("tmdb_rating_raw"),
            item.get("tmdb_vote_count"),
            item.get("public_rating_adjusted"),
            item.get("raw_like_probability"),
            item.get("like_probability"),
            item.get("interest_score"),
            item.get("recommendation_score"),
            item.get("confidence"),
            item.get("model_coverage"),
            item.get("personal_engine"),
            item.get("safe_score"),
            item.get("rank_score"),
            ",".join(item.get("sources", [])),
        )
    return path

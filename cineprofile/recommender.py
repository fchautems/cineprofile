from __future__ import annotations

import json
import math
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np

from . import __version__
from .candidate_pool import (
    SOURCE_POPULARITY,
    SOURCE_SEMANTIC,
    build_candidate_pool,
    favorite_seeds as _favorite_seeds,  # noqa: F401
    passes_date_filter as _passes_date_filter,  # noqa: F401
    personalize_candidate_order,
    selected_source_counts,
    vote_threshold as _vote_threshold,  # noqa: F401
)
from .db import connect, initialize, transaction
from .diagnostics import (
    build_search_diagnostic,
    configure_logging,
    new_search_id,
    save_search_diagnostic,
)
from . import hybrid_model as hm
from .personal_model import (
    ensure_personal_model,
    predict_personal_candidate,
)
from .preferences import load_feedback, load_preferences
from .public_rating import public_rating
from .ranking import exploration_for_mode, rerank_recommendations
from .recommendation_state import save_recommendation_state
from .semantic import semantic_evidence
from .tmdb import TmdbClient, enrich_candidates
from .watch_interest import score_watch_interest


RECOMMENDATION_PROTOCOL = 14
PERSONAL_RANKER_VERSION = "cineprofile-local-ranker-0.11.0"
LOCAL_NEIGHBOR_WEIGHT = 0.65
GLOBAL_MODEL_WEIGHT = 0.35
MAX_PUBLIC_RANK_WEIGHT = 0.10

AFFINITY_WEIGHTS = {
    "genres": 0.10,
    "keywords": 0.20,
    "people": 0.15,
    "semantic": 0.40,
    "explicit": 0.15,
}

PEOPLE_WEIGHTS = {
    "directors": 0.40,
    "writers": 0.25,
    "actors": 0.20,
    "cinematographers": 0.05,
    "composers": 0.05,
    "editors": 0.05,
}

PERSON_MIN_SUPPORT = {
    "directors": 2,
    "writers": 2,
    "actors": 5,
    "cinematographers": 3,
    "composers": 3,
    "editors": 3,
}

SEARCH_DEPTHS = {
    "Rapide": {
        "discover_pages": 6,
        "quality_pages": 2,
        "seed_count": 6,
        "recommendation_pages": 1,
        "similar_pages": 1,
        "creator_count": 3,
        "actor_count": 2,
        "keyword_count": 4,
        "genre_count": 2,
        "analysis_limit": 140,
    },
    "Normale": {
        "discover_pages": 20,
        "quality_pages": 4,
        "seed_count": 12,
        "recommendation_pages": 1,
        "similar_pages": 1,
        "creator_count": 8,
        "actor_count": 4,
        "keyword_count": 10,
        "genre_count": 5,
        "analysis_limit": 300,
    },
    "Approfondie": {
        "discover_pages": 35,
        "quality_pages": 7,
        "seed_count": 24,
        "recommendation_pages": 2,
        "similar_pages": 2,
        "creator_count": 16,
        "actor_count": 8,
        "keyword_count": 18,
        "genre_count": 8,
        "analysis_limit": 500,
    },
}


def _blend_personal_prediction_v09(
    learned_prediction: dict | None,
    semantic_item: dict,
    *,
    baseline: float,
    explicit: tuple[float, float] | None,
) -> dict | None:
    """Combine local liked/disliked neighbours with one global taste model.

    The public score is deliberately absent from this blend.  It remains
    available for reliability checks and, in exploratory modes only, as a
    small quality tie-breaker.
    """

    local_score = semantic_item.get("score")
    local_rating = semantic_item.get("predicted_rating")
    local_confidence = float(semantic_item.get("confidence") or 0.0)
    parts: list[tuple[float, float, float, float]] = []
    if local_score is not None:
        local_weight = (
            LOCAL_NEIGHBOR_WEIGHT
            if semantic_item.get("engine") == "semantic"
            else 0.45
        )
        parts.append(
            (
                local_weight,
                float(local_score),
                float(local_rating or baseline),
                local_confidence,
            )
        )
    if learned_prediction is not None:
        parts.append(
            (
                GLOBAL_MODEL_WEIGHT,
                float(learned_prediction["like_probability"]) / 100.0,
                float(learned_prediction["predicted_rating"]),
                float(learned_prediction["confidence"]) / 100.0,
            )
        )
    if not parts:
        return None
    total_weight = sum(row[0] for row in parts)
    probability = sum(row[0] * row[1] for row in parts) / total_weight
    predicted_rating = sum(row[0] * row[2] for row in parts) / total_weight
    confidence = sum(row[0] * row[3] for row in parts) / total_weight
    raw_probability = probability
    if explicit is not None:
        adjustment = (float(explicit[0]) - 0.5) * 0.30
        probability = max(0.01, min(0.99, probability + adjustment))
        predicted_rating = max(
            1.0,
            min(10.0, predicted_rating + 3.0 * adjustment),
        )
    half_interval = 0.65 + 1.35 * (1.0 - confidence)
    public_baseline = (
        learned_prediction.get("public_baseline_rating")
        if learned_prediction is not None
        else None
    )
    return {
        "like_probability": round(100.0 * probability, 1),
        "raw_like_probability": round(100.0 * raw_probability, 1),
        "predicted_rating": round(predicted_rating, 2),
        "prediction_low": round(
            max(1.0, predicted_rating - half_interval),
            1,
        ),
        "prediction_high": round(
            min(10.0, predicted_rating + half_interval),
            1,
        ),
        "confidence": round(100.0 * confidence, 1),
        "coverage": round(100.0 * confidence, 1),
        "public_baseline_rating": public_baseline,
        "public_raw_rating": (
            learned_prediction.get("public_raw_rating")
            if learned_prediction is not None
            else None
        ),
        "public_rating_reliability": (
            learned_prediction.get("public_rating_reliability")
            if learned_prediction is not None
            else 0.0
        ),
        "user_baseline_rating": round(baseline, 2),
        "personal_adjustment": round(predicted_rating - baseline, 2),
        "positive_signals": (
            learned_prediction.get("positive_signals", [])
            if learned_prediction is not None
            else []
        ),
        "negative_signals": (
            learned_prediction.get("negative_signals", [])
            if learned_prediction is not None
            else []
        ),
        "model_version": PERSONAL_RANKER_VERSION,
        "engine": "personal_v09",
        "variant": "local_global",
        "variant_label": "Voisins personnels + profil global",
        "underlying_engine": (
            learned_prediction.get("engine")
            if learned_prediction is not None
            else None
        ),
        "local_like_probability": (
            round(100.0 * float(local_score), 1)
            if local_score is not None
            else None
        ),
        "global_like_probability": (
            learned_prediction.get("like_probability")
            if learned_prediction is not None
            else None
        ),
        "local_neighbor_weight": (
            round(parts[0][0] / total_weight, 3)
            if local_score is not None
            else 0.0
        ),
        "public_influence_weight": 0.0,
        "base_like_rate": float(
            semantic_item.get("base_like_rate") or 0.20
        ),
        "positive_island": (
            learned_prediction.get("positive_island")
            if learned_prediction is not None
            else None
        ),
        "negative_island": (
            learned_prediction.get("negative_island")
            if learned_prediction is not None
            else None
        ),
        "island_margin": (
            learned_prediction.get("island_margin")
            if learned_prediction is not None
            else None
        ),
        "positive_similarity": (
            learned_prediction.get("positive_similarity")
            if learned_prediction is not None
            else None
        ),
        "negative_similarity": (
            learned_prediction.get("negative_similarity")
            if learned_prediction is not None
            else None
        ),
    }


def _dimension_map(profile: dict, key: str) -> dict[str, dict]:
    return {
        str(row["id"]): row
        for row in profile.get("dimensions", {}).get(key, [])
        if "id" in row
    }


def _dimension_evidence(rows: Iterable[dict]) -> tuple[float, float] | None:
    observations = list(rows)
    if not observations:
        return None
    # Le nombre de films sert déjà à réduire l'affinité lors du profilage.
    # Il ne doit pas faire dominer un genre simplement parce qu'il est fréquent.
    weights = np.array(
        [max(0.05, float(row.get("confidence", 0.0))) for row in observations],
        dtype=float,
    )
    signals = np.array(
        [float(row.get("affinity", 0.0)) for row in observations],
        dtype=float,
    )
    signal = float(np.average(signals, weights=weights))
    score = 0.5 + 0.5 * math.tanh(signal)
    confidence = min(
        1.0,
        float(np.mean([row.get("confidence", 0.0) for row in observations]))
        * (1.0 + 0.10 * math.log1p(len(observations))),
    )
    return score, confidence


def _weighted_available(
    values: dict[str, tuple[float, float] | None],
    weights: dict[str, float],
) -> tuple[float, float]:
    available = {key: value for key, value in values.items() if value is not None}
    if not available:
        return 0.5, 0.0
    available_weight = sum(weights[key] for key in available)
    score = (
        sum(weights[key] * available[key][0] for key in available)
        / available_weight
    )
    confidence = (
        sum(weights[key] * available[key][1] for key in available)
        / sum(weights.values())
    )
    return float(score), min(1.0, float(confidence))


def _candidate_entities(candidate: dict) -> dict[str, list[tuple[str, str]]]:
    genres = [
        (str(item["id"]), item["name"]) for item in candidate.get("genres", [])
    ]
    keywords_payload = candidate.get("keywords", {})
    keywords = keywords_payload.get(
        "keywords",
        keywords_payload.get("results", []),
    )
    people: dict[str, list[tuple[str, str]]] = {
        "directors": [],
        "writers": [],
        "actors": [],
        "cinematographers": [],
        "composers": [],
        "editors": [],
    }
    # Seuls les rôles principaux peuvent caractériser le style d'un film.
    # Les figurants et petits rôles ne doivent pas créer une fausse affinité.
    for actor in candidate.get("credits", {}).get("cast", [])[:5]:
        people["actors"].append((str(actor["id"]), actor["name"]))
    jobs = {
        "Director": "directors",
        "Screenplay": "writers",
        "Writer": "writers",
        "Story": "writers",
        "Director of Photography": "cinematographers",
        "Original Music Composer": "composers",
        "Editor": "editors",
    }
    for crew in candidate.get("credits", {}).get("crew", []):
        dimension = jobs.get(crew.get("job", ""))
        if dimension:
            people[dimension].append((str(crew["id"]), crew["name"]))
    return {
        "genres": genres,
        "keywords": [
            (str(item["id"]), item["name"]) for item in keywords
        ],
        **people,
    }


def _explicit_evidence(
    entities: dict[str, list[tuple[str, str]]],
    preferences: dict[tuple[str, str], dict],
) -> tuple[tuple[float, float] | None, list[str], bool]:
    adjustments: list[tuple[int, str]] = []
    excluded = False
    for entity_type, items in entities.items():
        for entity_id, name in items:
            preference = preferences.get((entity_type, entity_id))
            if not preference:
                continue
            value = int(preference["adjustment"])
            adjustments.append((value, name))
            if value == -2:
                excluded = True
    if not adjustments:
        return None, [], False
    value = float(np.mean([item[0] for item in adjustments]))
    score = max(0.0, min(1.0, 0.5 + 0.22 * value))
    labels = [
        f"{name} ({'favorisé' if adjustment > 0 else 'réduit'})"
        for adjustment, name in adjustments
        if adjustment != -2
    ]
    return (score, 1.0), labels, excluded


def _freshness(release_date: str | None) -> float:
    if not release_date:
        return 0.3
    try:
        released = date.fromisoformat(release_date)
    except ValueError:
        return 0.3
    age_days = max(0, (date.today() - released).days)
    return math.exp(-age_days / 730)


def _quality_metrics(candidate: dict) -> tuple[float, float]:
    evidence = public_rating(
        candidate.get("vote_average"),
        candidate.get("vote_count"),
        source="tmdb",
    )
    adjusted = (
        float(evidence.adjusted_rating)
        if evidence.adjusted_rating is not None
        else evidence.prior_mean
    )
    score = max(0.0, min(1.0, (adjusted - 4.5) / 4.0))
    return score, adjusted


def _principal_credits(candidate: dict) -> tuple[list[str], list[str], list[str]]:
    cast = [
        item["name"]
        for item in candidate.get("credits", {}).get("cast", [])[:4]
        if item.get("name")
    ]
    directors: list[str] = []
    writers: list[str] = []
    for crew in candidate.get("credits", {}).get("crew", []):
        name = crew.get("name")
        if not name:
            continue
        if crew.get("job") == "Director" and name not in directors:
            directors.append(name)
        if (
            crew.get("job") in {"Screenplay", "Writer", "Story"}
            and name not in writers
        ):
            writers.append(name)
    return directors, cast, writers


def _has_excluded_genre(
    candidate: dict,
    excluded_genre_ids: set[int] | None,
) -> bool:
    if not excluded_genre_ids:
        return False
    ids = {
        int(value)
        for value in candidate.get("genre_ids", [])
        if str(value).isdigit()
    }
    ids.update(
        int(item["id"])
        for item in candidate.get("genres", [])
        if isinstance(item, dict) and str(item.get("id", "")).isdigit()
    )
    return bool(ids & excluded_genre_ids)


def score_candidates(
    profile: dict,
    candidates: list[dict],
    database: str | Path | None = None,
    *,
    exploration: int | None = None,
    ranking_mode: str = "Valeurs sûres",
    semantic_enabled: bool = False,
    region: str = "CH",
    excluded_genre_ids: set[int] | None = None,
) -> list[dict]:
    baseline = float(profile["summary"]["average_rating"])
    maps = {
        key: _dimension_map(profile, key)
        for key in (
            "genres",
            "keywords",
            "directors",
            "writers",
            "actors",
            "cinematographers",
            "composers",
            "editors",
        )
    }
    preferences = load_preferences(database)
    hybrid_state = hm.ensure_hybrid_model(database)
    hybrid_predictions: dict[int, dict] = {}
    personal_model_state = None
    if hybrid_state.status == "ready" and hybrid_state.model is not None:
        try:
            predictions = hm.predict_hybrid_candidates(
                database,
                hybrid_state.model,
                candidates,
            )
            hybrid_predictions = {
                int(candidate["id"]): prediction
                for candidate, prediction in zip(
                    candidates,
                    predictions,
                    strict=True,
                )
            }
        except Exception:
            configure_logging(database).exception(
                "hybrid_candidate_prediction_failed | fallback=linear_v06"
            )
            hybrid_predictions = {}
    if not hybrid_predictions:
        personal_model_state = ensure_personal_model(database)
    semantic = semantic_evidence(
        database,
        candidates,
        baseline,
        enabled=semantic_enabled,
    )

    scored: list[dict] = []
    for candidate in candidates:
        if _has_excluded_genre(candidate, excluded_genre_ids):
            continue
        entities = _candidate_entities(candidate)
        dimension_evidence: dict[str, tuple[float, float] | None] = {}
        matched_details: list[dict] = []
        for key, items in entities.items():
            rows = []
            for entity_id, name in items:
                row = maps.get(key, {}).get(entity_id)
                if row:
                    minimum_support = PERSON_MIN_SUPPORT.get(key, 1)
                    if int(row.get("seen", 0)) < minimum_support:
                        continue
                    rows.append(row)
                    matched_details.append(
                        {
                            "dimension": key,
                            "name": name,
                            "affinity": row["affinity"],
                            "seen": row["seen"],
                            "average_rating": row["average_user_rating"],
                        }
                    )
            dimension_evidence[key] = _dimension_evidence(rows)

        people_score, people_confidence = _weighted_available(
            {key: dimension_evidence[key] for key in PEOPLE_WEIGHTS},
            PEOPLE_WEIGHTS,
        )
        semantic_item = semantic.get(
            int(candidate["id"]),
            {
                "score": None,
                "predicted_rating": None,
                "base_like_rate": 0.20,
                "confidence": 0.0,
                "similarity": 0.0,
                "positive_similarity": 0.0,
                "negative_similarity": 0.0,
                "neighbors": [],
                "engine": "none",
            },
        )
        explicit, explicit_labels, excluded = _explicit_evidence(
            entities,
            preferences,
        )
        if excluded:
            continue

        evidence_score, confidence = _weighted_available(
            {
                "genres": dimension_evidence["genres"],
                "keywords": dimension_evidence["keywords"],
                "people": (
                    (people_score, people_confidence)
                    if people_confidence > 0
                    else None
                ),
                "semantic": (
                    (
                        float(semantic_item["score"]),
                        float(semantic_item["confidence"]),
                    )
                    if semantic_item["score"] is not None
                    else None
                ),
                "explicit": explicit,
            },
            AFFINITY_WEIGHTS,
        )
        personal_score = 0.5 + (evidence_score - 0.5) * (
            0.25 + 0.75 * confidence
        )
        base_prediction = hybrid_predictions.get(int(candidate["id"]))
        if (
            base_prediction is None
            and personal_model_state is not None
            and personal_model_state.model is not None
            and personal_model_state.summary.get("active_engine")
            in {"linear_v06", "islands_v07"}
        ):
            base_prediction = predict_personal_candidate(
                personal_model_state.model,
                candidate,
            )
        learned_prediction = _blend_personal_prediction_v09(
            base_prediction,
            semantic_item,
            baseline=baseline,
            explicit=explicit,
        )
        if learned_prediction is not None:
            personal_score = float(
                learned_prediction["like_probability"]
            ) / 100.0
            confidence = float(learned_prediction["confidence"]) / 100.0
        base_like_rate = float(
            learned_prediction.get("base_like_rate", 0.20)
            if learned_prediction
            else 0.20
        )
        neighbors = semantic_item.get("neighbors", [])
        matched_details.sort(
            key=lambda row: (float(row["affinity"]), int(row["seen"])),
            reverse=True,
        )
        interest = score_watch_interest(
            candidate,
            profile=profile,
            entities=entities,
            matched_details=matched_details,
            semantic_neighbors=neighbors,
            preferences=preferences,
            like_probability=personal_score,
            base_like_rate=base_like_rate,
        )
        recommendation_score = (
            float(interest["recommendation_score"]) / 100.0
        )
        quality_score, bayesian_rating = _quality_metrics(candidate)
        freshness_score = _freshness(candidate.get("release_date"))
        exploration_value = exploration_for_mode(ranking_mode, exploration)
        exploration_ratio = exploration_value / 100.0
        novelty_score = 1.0 - min(
            1.0,
            max(0.0, float(semantic_item.get("similarity", 0.0))),
        )
        safe_rank = recommendation_score + 0.001 * confidence
        discovery_rank = (
            0.70 * recommendation_score
            + MAX_PUBLIC_RANK_WEIGHT * quality_score
            + 0.10 * freshness_score
            + 0.10 * novelty_score
        )
        rank_score = (
            (1.0 - exploration_ratio) * safe_rank
            + exploration_ratio * discovery_rank
        )
        affinity_index = round(100 * personal_score, 1)
        confidence_label = (
            "forte"
            if confidence >= 0.66
            else "moyenne"
            if confidence >= 0.34
            else "faible"
        )

        reasons: list[str] = []
        if interest["positive_reasons"]:
            reasons.append(
                "Pourquoi il peut donner envie : "
                + ", ".join(
                    row["label"]
                    for row in interest["positive_reasons"][:3]
                )
            )
        if interest["reservations"]:
            reasons.append(
                "Freins possibles : "
                + ", ".join(
                    row["label"] for row in interest["reservations"][:3]
                )
            )
        if learned_prediction is not None:
            reasons.append(
                "Note personnelle prévue : "
                f"{learned_prediction['predicted_rating']:.1f}/10"
            )
            probability_reduction = float(
                learned_prediction.get("raw_like_probability")
                or learned_prediction["like_probability"]
            ) - float(learned_prediction["like_probability"])
            if probability_reduction >= 8.0:
                reasons.append(
                    "Estimation rendue prudente par le nombre de votes ou "
                    "les informations disponibles"
                )
        if explicit_labels:
            reasons.append("Tes réglages : " + ", ".join(explicit_labels[:3]))
        if learned_prediction and learned_prediction["positive_signals"]:
            reasons.append(
                "Signaux appris : "
                + ", ".join(
                    row["label"]
                    for row in learned_prediction["positive_signals"][:3]
                )
            )
        if matched_details:
            reasons.append(
                "Affinités connues : "
                + ", ".join(row["name"] for row in matched_details[:4])
            )
        positive_similarity = float(
            semantic_item.get("positive_similarity", 0.0)
        )
        negative_similarity = float(
            semantic_item.get("negative_similarity", 0.0)
        )
        if neighbors:
            reasons.append(
                "Histoires proches : "
                + ", ".join(
                    f"{row['title']}"
                    + (
                        f" ({row['rating']:g}/10)"
                        if row.get("rating") is not None
                        else " (pas intéressé)"
                    )
                    for row in neighbors[:3]
                )
            )
        if (
            negative_similarity >= 0.45
            and negative_similarity > positive_similarity + 0.05
        ):
            reasons.append(
                "Réserve : histoire proche de films moins aimés ou refusés"
            )
        if quality_score >= 0.65:
            reasons.append(
                "Qualité publique suffisante, utilisée seulement comme garde-fou"
            )
        if not str(candidate.get("overview") or "").strip():
            reasons.append("Résumé absent : confiance volontairement réduite")
        if not reasons:
            reasons.append("Découverte hors des habitudes principales")

        providers = (
            candidate.get("watch/providers", {})
            .get("results", {})
            .get(region, {})
        )
        provider_summary = {
            access: [item["provider_name"] for item in providers.get(access, [])]
            for access in ("flatrate", "free", "ads", "rent", "buy")
            if providers.get(access)
        }
        directors, cast, writers = _principal_credits(candidate)
        match_tier = (
            "Correspondance solide"
            if (
                recommendation_score >= 0.68
                and float(interest["interest_score"]) >= 62.0
                and personal_score >= base_like_rate * 1.30
                and confidence >= 0.45
            )
            else "Piste plausible"
            if recommendation_score >= 0.52
            else "Solution de repli"
        )
        negative_genres = [
            name
            for entity_id, name in entities.get("genres", [])
            if (
                (row := maps.get("genres", {}).get(entity_id))
                and int(row.get("seen", 0)) >= 3
                and float(row.get("affinity", 0.0)) < 0.0
            )
        ]
        scored.append(
            {
                "tmdb_id": int(candidate["id"]),
                "imdb_id": candidate.get("external_ids", {}).get("imdb_id"),
                "title": candidate.get("title") or candidate.get("name"),
                "release_date": candidate.get("release_date"),
                "score": affinity_index,
                "affinity_index": affinity_index,
                "like_probability": affinity_index,
                "predicted_rating": (
                    learned_prediction["predicted_rating"]
                    if learned_prediction
                    else None
                ),
                "prediction_low": (
                    learned_prediction["prediction_low"]
                    if learned_prediction
                    else None
                ),
                "prediction_high": (
                    learned_prediction["prediction_high"]
                    if learned_prediction
                    else None
                ),
                "personal_adjustment": (
                    learned_prediction["personal_adjustment"]
                    if learned_prediction
                    else None
                ),
                "user_baseline_rating": (
                    learned_prediction.get("user_baseline_rating")
                    if learned_prediction
                    else round(baseline, 2)
                ),
                "public_rating_adjusted": (
                    learned_prediction.get("public_baseline_rating")
                    if learned_prediction
                    else round(bayesian_rating, 2)
                ),
                "public_rating_reliability": (
                    learned_prediction.get("public_rating_reliability")
                    if learned_prediction
                    else round(
                        100.0
                        * public_rating(
                            candidate.get("vote_average"),
                            candidate.get("vote_count"),
                            source="tmdb",
                        ).reliability,
                        1,
                    )
                ),
                "raw_like_probability": (
                    learned_prediction.get("raw_like_probability")
                    if learned_prediction
                    else affinity_index
                ),
                "personal_model_used": learned_prediction is not None,
                "personal_model_version": (
                    learned_prediction["model_version"]
                    if learned_prediction
                    else None
                ),
                "personal_engine": (
                    learned_prediction.get("engine")
                    if learned_prediction
                    else "legacy_v05"
                ),
                "personal_variant_label": (
                    learned_prediction.get("variant_label")
                    if learned_prediction
                    else None
                ),
                "underlying_personal_engine": (
                    learned_prediction.get("underlying_engine")
                    if learned_prediction
                    else None
                ),
                "local_neighbor_weight": (
                    learned_prediction.get("local_neighbor_weight")
                    if learned_prediction
                    else None
                ),
                "local_like_probability": (
                    learned_prediction.get("local_like_probability")
                    if learned_prediction
                    else None
                ),
                "global_like_probability": (
                    learned_prediction.get("global_like_probability")
                    if learned_prediction
                    else None
                ),
                "public_influence_weight": (
                    learned_prediction.get("public_influence_weight")
                    if learned_prediction
                    else None
                ),
                "base_like_rate": base_like_rate,
                "interest_model_version": interest["model_version"],
                "interest_score": interest["interest_score"],
                "interest_label": interest["interest_label"],
                "interest_confidence": interest["interest_confidence"],
                "interest_positive_reasons": interest["positive_reasons"],
                "interest_reservations": interest["reservations"],
                "satisfaction_lift_index": interest[
                    "satisfaction_lift_index"
                ],
                "recommendation_score": interest[
                    "recommendation_score"
                ],
                "base_like_rate_percent": interest["base_like_rate"],
                "like_probability_lift_points": interest[
                    "like_probability_lift_points"
                ],
                "like_probability_lift_ratio": interest[
                    "like_probability_lift_ratio"
                ],
                "match_tier": match_tier,
                "positive_island": (
                    learned_prediction.get("positive_island")
                    if learned_prediction
                    else None
                ),
                "negative_island": (
                    learned_prediction.get("negative_island")
                    if learned_prediction
                    else None
                ),
                "island_margin": (
                    learned_prediction.get("island_margin")
                    if learned_prediction
                    else None
                ),
                "positive_similarity": (
                    learned_prediction.get("positive_similarity")
                    if learned_prediction
                    else None
                ),
                "negative_similarity": (
                    learned_prediction.get("negative_similarity")
                    if learned_prediction
                    else None
                ),
                "model_coverage": (
                    learned_prediction["coverage"]
                    if learned_prediction
                    else None
                ),
                "learned_positive_signals": (
                    learned_prediction["positive_signals"]
                    if learned_prediction
                    else []
                ),
                "learned_negative_signals": (
                    learned_prediction["negative_signals"]
                    if learned_prediction
                    else []
                ),
                "confidence": round(100 * confidence, 1),
                "confidence_label": confidence_label,
                "rank_score": round(100 * rank_score, 3),
                "safe_score": round(100 * safe_rank, 3),
                "discovery_score": round(100 * discovery_rank, 3),
                "reasons": reasons,
                "poster_path": candidate.get("poster_path"),
                "overview": candidate.get("overview"),
                "vote_average": candidate.get("vote_average"),
                "vote_count": candidate.get("vote_count"),
                "bayesian_rating": round(bayesian_rating, 2),
                "genres": [
                    item["name"] for item in candidate.get("genres", [])
                ],
                "keywords": [
                    name
                    for _, name in entities.get("keywords", [])
                ],
                "original_language": candidate.get("original_language"),
                "runtime_minutes": candidate.get("runtime"),
                "providers_ch": provider_summary,
                "directors": directors,
                "cast": cast,
                "writers": writers,
                "collection_id": (
                    candidate.get("belongs_to_collection", {}).get("id")
                    if isinstance(
                        candidate.get("belongs_to_collection"),
                        dict,
                    )
                    else None
                ),
                "collection_name": (
                    candidate.get("belongs_to_collection", {}).get("name")
                    if isinstance(
                        candidate.get("belongs_to_collection"),
                        dict,
                    )
                    else None
                ),
                "sources": candidate.get("_sources", []),
                "retrieval_score": candidate.get("_retrieval_score"),
                "retrieval_confidence": candidate.get(
                    "_retrieval_confidence"
                ),
                "negative_genres": negative_genres,
                "matched_details": matched_details[:12],
                "semantic_neighbors": neighbors,
                "semantic_engine": semantic_item.get("engine", "none"),
                "semantic_error": semantic_item.get("error"),
                "components": {
                    "genres": (
                        round(dimension_evidence["genres"][0], 3)
                        if dimension_evidence["genres"]
                        else None
                    ),
                    "keywords": (
                        round(dimension_evidence["keywords"][0], 3)
                        if dimension_evidence["keywords"]
                        else None
                    ),
                    "people": (
                        round(float(people_score), 3)
                        if people_confidence > 0
                        else None
                    ),
                    "semantic": (
                        round(float(semantic_item["score"]), 3)
                        if semantic_item["score"] is not None
                        else None
                    ),
                    "semantic_positive_similarity": round(
                        positive_similarity,
                        3,
                    ),
                    "semantic_negative_similarity": round(
                        negative_similarity,
                        3,
                    ),
                    "explicit": (
                        round(float(explicit[0]), 3) if explicit else None
                    ),
                    "quality": round(quality_score, 3),
                    "public_rank_weight_max": MAX_PUBLIC_RANK_WEIGHT,
                    "freshness": round(freshness_score, 3),
                    "novelty": round(novelty_score, 3),
                    "interest": round(
                        float(interest["interest_score"]) / 100.0,
                        3,
                    ),
                    "satisfaction_lift": round(
                        float(interest["satisfaction_lift_index"]) / 100.0,
                        3,
                    ),
                    "recommendation": round(
                        recommendation_score,
                        3,
                    ),
                },
            }
        )
    return rerank_recommendations(
        scored,
        mode=ranking_mode,
        custom_exploration=exploration,
    )


def _candidate_pool(
    client: TmdbClient,
    profile: dict,
    database: str | Path | None,
    *,
    start_date: str | None,
    end_date: str | None,
    depth: str,
    reliability: str,
    excluded_genre_ids: set[int] | None = None,
) -> tuple[list[dict], dict[str, int], dict[str, object]]:
    return build_candidate_pool(
        client,
        profile,
        database,
        start_date=start_date,
        end_date=end_date,
        settings=SEARCH_DEPTHS[depth],
        reliability=reliability,
        excluded_genre_ids=excluded_genre_ids,
        excluded_genre=_has_excluded_genre,
    )


def _enrich_cached(
    client: TmdbClient,
    candidates: list[dict],
    database: str | Path | None,
    *,
    limit: int,
) -> tuple[list[dict], int, int]:
    selected = candidates[:limit]
    if not selected:
        return [], 0, 0
    language = getattr(client, "language", "fr-FR")
    region = getattr(client, "region", "CH")
    ids = [int(item["id"]) for item in selected]
    placeholders = ",".join("?" for _ in ids)
    with connect(database) as connection:
        rows = connection.execute(
            f"""
            SELECT tmdb_id, payload_json, fetched_at
            FROM candidate_cache
            WHERE language=? AND region=? AND tmdb_id IN ({placeholders})
            """,
            [language, region, *ids],
        ).fetchall()
    cutoff = datetime.now(UTC) - timedelta(days=30)
    cached: dict[int, dict] = {}
    for row in rows:
        try:
            fetched = datetime.fromisoformat(row["fetched_at"])
            if fetched >= cutoff:
                cached[int(row["tmdb_id"])] = json.loads(row["payload_json"])
        except (ValueError, TypeError, json.JSONDecodeError):
            continue

    enriched: list[dict] = []
    misses: list[dict] = []
    selected_by_id = {int(item["id"]): item for item in selected}
    for candidate_id in ids:
        if candidate_id in cached:
            details = cached[candidate_id]
            details["_discover"] = selected_by_id[candidate_id]
            source = selected_by_id[candidate_id]
            details["_sources"] = source.get("_sources", [])
            for key in (
                "_retrieval_score",
                "_retrieval_confidence",
                "_retrieval_utility",
            ):
                details[key] = source.get(key)
            enriched.append(details)
        else:
            misses.append(selected_by_id[candidate_id])

    downloaded = enrich_candidates(client, misses)
    now = datetime.now(UTC).isoformat()
    if downloaded:
        with transaction(database) as connection:
            connection.executemany(
                """
                INSERT INTO candidate_cache(
                  tmdb_id, language, region, payload_json, fetched_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(tmdb_id, language, region) DO UPDATE SET
                  payload_json=excluded.payload_json,
                  fetched_at=excluded.fetched_at
                """,
                [
                    (
                        int(item["id"]),
                        language,
                        region,
                        json.dumps(
                            {
                                key: value
                                for key, value in item.items()
                                if key not in {"_discover", "_sources"}
                            },
                            ensure_ascii=False,
                        ),
                        now,
                    )
                    for item in downloaded
                ],
            )
    for details in downloaded:
        candidate = selected_by_id[int(details["id"])]
        details["_sources"] = candidate.get("_sources", [])
        for key in (
            "_retrieval_score",
            "_retrieval_confidence",
            "_retrieval_utility",
        ):
            details[key] = candidate.get(key)
        enriched.append(details)
    order = {candidate_id: index for index, candidate_id in enumerate(ids)}
    enriched.sort(key=lambda item: order[int(item["id"])])
    return enriched, len(cached), len(downloaded)


def _persist_recommendations(
    profile: dict,
    results: list[dict],
    database: str | Path | None,
) -> None:
    run_id = profile.get("profile_run_id")
    if not run_id:
        return
    created_at = datetime.now(UTC).isoformat()
    with transaction(database) as connection:
        connection.execute(
            "DELETE FROM recommendations WHERE profile_run_id=?",
            (run_id,),
        )
        connection.executemany(
            """
            INSERT INTO recommendations(
              profile_run_id, tmdb_id, imdb_id, title, release_date,
              score, reasons_json, payload_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    run_id,
                    item["tmdb_id"],
                    item.get("imdb_id"),
                    item["title"],
                    item.get("release_date"),
                    item["score"],
                    json.dumps(item["reasons"], ensure_ascii=False),
                    json.dumps(item, ensure_ascii=False),
                    created_at,
                )
                for item in results
            ],
        )


def recommend_movies(
    client: TmdbClient,
    profile: dict,
    database: str | Path | None = None,
    *,
    start_date: str | None,
    end_date: str | None,
    depth: str = "Normale",
    reliability: str = "Forte",
    ranking_mode: str = "Valeurs sûres",
    exploration: int | None = None,
    semantic_enabled: bool = True,
    analysis_limit: int | None = None,
    excluded_genre_ids: set[int] | None = None,
) -> tuple[list[dict], dict[str, object]]:
    initialize(database)
    if depth not in SEARCH_DEPTHS:
        raise ValueError(f"Profondeur inconnue : {depth}")
    search_id = new_search_id()
    settings_payload = {
        "start_date": start_date,
        "end_date": end_date,
        "depth": depth,
        "reliability": reliability,
        "ranking_mode": ranking_mode,
        "exploration": exploration,
        "semantic_enabled": semantic_enabled,
        "analysis_limit": analysis_limit,
        "excluded_genre_ids": sorted(excluded_genre_ids or set()),
    }
    logger = configure_logging(database)
    logger.info(
        "search_started | search_id=%s | settings=%s",
        search_id,
        json.dumps(settings_payload, ensure_ascii=False, sort_keys=True),
    )
    candidates, source_counts, pool_diagnostics = _candidate_pool(
        client,
        profile,
        database,
        start_date=start_date,
        end_date=end_date,
        depth=depth,
        reliability=reliability,
        excluded_genre_ids=excluded_genre_ids,
    )
    feedback = load_feedback(database)
    with connect(database) as connection:
        seen_tmdb = {
            int(row[0])
            for row in connection.execute(
                "SELECT tmdb_id FROM titles WHERE tmdb_id IS NOT NULL"
            )
        }
        seen_imdb = {
            row[0] for row in connection.execute("SELECT imdb_id FROM titles")
        }
    excluded_feedback = {
        tmdb_id
        for tmdb_id, row in feedback.items()
        if row["action"] in {"not_interested", "already_seen"}
    }
    not_seen_on_tmdb = [
        candidate
        for candidate in candidates
        if int(candidate["id"]) not in seen_tmdb
    ]
    unseen = [
        candidate
        for candidate in not_seen_on_tmdb
        if int(candidate["id"]) not in excluded_feedback
    ]
    limit = analysis_limit or SEARCH_DEPTHS[depth]["analysis_limit"]
    retrieval_evidence = semantic_evidence(
        database,
        unseen,
        float(profile["summary"]["average_rating"]),
        enabled=semantic_enabled,
    )
    unseen, semantic_source_count = personalize_candidate_order(
        unseen,
        retrieval_evidence,
        maximum_semantic_source=max(40, limit // 2),
    )
    if semantic_source_count:
        source_counts[SOURCE_SEMANTIC] = semantic_source_count
    analysis_source_counts = selected_source_counts(unseen, limit)
    selected_candidates = unseen[:limit]
    popularity_only_selected = sum(
        candidate.get("_sources") == [SOURCE_POPULARITY]
        for candidate in selected_candidates
    )
    enriched, cache_hits, downloaded = _enrich_cached(
        client,
        unseen,
        database,
        limit=limit,
    )
    unseen_enriched = [
        item
        for item in enriched
        if item.get("external_ids", {}).get("imdb_id") not in seen_imdb
    ]
    scored = score_candidates(
        profile,
        unseen_enriched,
        database,
        ranking_mode=ranking_mode,
        exploration=exploration,
        semantic_enabled=semantic_enabled,
        region=getattr(client, "region", "CH"),
        excluded_genre_ids=excluded_genre_ids,
    )
    _persist_recommendations(profile, scored, database)
    semantic_engine = (
        scored[0].get("semantic_engine", "none") if scored else "none"
    )
    semantic_error = next(
        (
            item.get("semantic_error")
            for item in scored
            if item.get("semantic_error")
        ),
        None,
    )
    diagnostics: dict[str, object] = {
        "search_id": search_id,
        "settings": settings_payload,
        "window_start": start_date or "Sans limite",
        "window_end": end_date or "Sans limite",
        "source_counts": source_counts,
        "unique_candidates": pool_diagnostics["raw_unique_candidates"],
        **pool_diagnostics,
        "excluded_already_seen_tmdb": len(candidates) - len(not_seen_on_tmdb),
        "excluded_by_feedback": len(not_seen_on_tmdb) - len(unseen),
        "selected_for_enrichment": min(len(unseen), limit),
        "selected_source_counts": analysis_source_counts,
        "semantic_retrieval_candidates": semantic_source_count,
        "popularity_only_selected": popularity_only_selected,
        "popularity_only_selected_share": round(
            popularity_only_selected / max(1, len(selected_candidates)),
            4,
        ),
        "cache_hits": cache_hits,
        "downloaded_details": downloaded,
        "enriched_successfully": len(enriched),
        "excluded_already_seen_imdb": len(enriched) - len(unseen_enriched),
        "scored": len(scored),
        "returned": len(scored),
        "semantic_engine": semantic_engine,
        "semantic_error": semantic_error,
        "personal_engine": (
            scored[0].get("personal_engine") if scored else None
        ),
        "personal_variant_label": (
            scored[0].get("personal_variant_label") if scored else None
        ),
        "active_model_configuration": hm.active_configuration(database),
        "adaptive_learning": {
            "rated_films": int(
                profile.get("summary", {}).get("rated_titles")
                or profile.get("summary", {}).get("rated_count")
                or 0
            ),
            "watchlist_signals": sum(
                row["action"] == "watchlist" for row in feedback.values()
            ),
            "negative_signals": sum(
                row["action"] == "not_interested"
                for row in feedback.values()
            ),
            "seen_exclusions": sum(
                row["action"] == "already_seen"
                for row in feedback.values()
            ),
            "rating_updates": (
                "automatic_model_fingerprint"
            ),
            "watchlist_role": "interest_only",
            "negative_role": "semantic_rejection",
            "seen_role": "exclusion_only",
        },
        "discovery_engine": (
            "personal_v09 + watch_interest_v011 + semantic_feedback + diversity"
        ),
        "excluded_genre_ids": sorted(excluded_genre_ids or set()),
    }
    diagnostic_payload = build_search_diagnostic(
        app_version=__version__,
        search_id=search_id,
        profile=profile,
        diagnostics=diagnostics,
        results=scored,
        settings=settings_payload,
    )
    diagnostic_path = save_search_diagnostic(
        database,
        diagnostic_payload,
    )
    diagnostics["diagnostic_path"] = str(diagnostic_path)
    diagnostics["diagnostic_file_name"] = diagnostic_path.name
    save_recommendation_state(
        profile.get("profile_run_id"),
        settings_payload,
        diagnostics,
        database,
    )
    return scored, diagnostics

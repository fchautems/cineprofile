from __future__ import annotations

from collections.abc import Iterable
from typing import Final

from .candidate_pool import SOURCE_BACK_CATALOG


RANKING_MODES = {
    "Valeurs sûres": {
        "exploration": 0,
        "diversity": 0.0,
        "description": (
            "Préserve la chance prudente tout en évitant qu’un même signal "
            "négatif occupe presque tout le haut de la liste."
        ),
    },
    "Équilibré": {
        "exploration": 30,
        "diversity": 0.10,
        "description": (
            "Conserve une forte affinité tout en évitant une liste trop "
            "répétitive."
        ),
    },
    "Découvertes": {
        "exploration": 65,
        "diversity": 0.22,
        "description": (
            "Ouvre davantage les histoires et équipes, sans promouvoir un "
            "film nettement moins sûr."
        ),
    },
}

SAFE_LIST_LABEL: Final = "Valeurs sûres"
DISCOVERY_LIST_LABEL: Final = "Découvertes pour toi"
CLASSIC_LIST_LABEL: Final = "Classiques à découvrir"
SAFE_TOP_EXCLUDED_FROM_DISCOVERIES: Final = 10
SAFE_PUBLIC_WEIGHT: Final = 0.70
SAFE_RECOMMENDATION_WEIGHT: Final = 0.15
SAFE_INTEREST_WEIGHT: Final = 0.15
SAFE_PRIMARY_PUBLIC_FLOOR: Final = 6.70
SAFE_PRIMARY_RECOMMENDATION_FLOOR: Final = 52.0
SAFE_PRIMARY_INTEREST_FLOOR: Final = 50.0
SAFE_SECONDARY_PUBLIC_FLOOR: Final = 6.50
SAFE_SECONDARY_RECOMMENDATION_FLOOR: Final = 48.0
SAFE_SECONDARY_INTEREST_FLOOR: Final = 45.0


def exploration_for_mode(
    mode: str,
    custom_exploration: int | None,
) -> int:
    if custom_exploration is not None:
        return max(0, min(100, int(custom_exploration)))
    if mode not in RANKING_MODES:
        raise ValueError(f"Mode de classement inconnu : {mode}")
    return int(RANKING_MODES[mode]["exploration"])


def _normalised(values: Iterable[object]) -> set[str]:
    return {
        str(value).strip().casefold()
        for value in values
        if str(value).strip()
    }


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def candidate_similarity(left: dict, right: dict) -> float:
    left_collection = left.get("collection_id")
    right_collection = right.get("collection_id")
    if (
        left_collection is not None
        and right_collection is not None
        and str(left_collection) == str(right_collection)
    ):
        return 1.0
    genres = _jaccard(
        _normalised(left.get("genres", [])),
        _normalised(right.get("genres", [])),
    )
    keywords = _jaccard(
        _normalised(left.get("keywords", [])),
        _normalised(right.get("keywords", [])),
    )
    directors = _jaccard(
        _normalised(left.get("directors", [])),
        _normalised(right.get("directors", [])),
    )
    cast = _jaccard(
        _normalised(left.get("cast", [])[:4]),
        _normalised(right.get("cast", [])[:4]),
    )
    return min(
        1.0,
        0.30 * genres
        + 0.30 * keywords
        + 0.25 * directors
        + 0.15 * cast,
    )


def _number(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _public_rating(row: dict) -> float:
    value = row.get("bayesian_rating")
    if value is None:
        value = row.get("public_rating_adjusted")
    return _number(value, -1.0)


def _has_personal_guardrails(row: dict) -> bool:
    return (
        row.get("recommendation_score") is not None
        and row.get("interest_score") is not None
    )


def safe_value_evidence(row: dict) -> dict[str, float | int | str]:
    """Combine public quality with a conservative personal compatibility gate.

    The public rating remains the dominant score.  Personal evidence is used
    to stop publicly acclaimed but clearly irrelevant films from occupying the
    top of the safe list.  Missing personal evidence deliberately falls back
    to the historical public-only order.
    """

    public_rating = _public_rating(row)
    public_quality = max(
        0.0,
        min(1.0, (public_rating - 6.0) / 2.0),
    )
    if not _has_personal_guardrails(row):
        return {
            "score": round(100.0 * public_quality, 3),
            "tier": 1,
            "label": "Qualité publique",
        }

    recommendation = max(
        0.0,
        min(100.0, _number(row.get("recommendation_score"))),
    )
    interest = max(
        0.0,
        min(100.0, _number(row.get("interest_score"))),
    )
    predicted = row.get("predicted_rating")
    baseline = row.get("user_baseline_rating")
    predicted_is_safe = (
        predicted is None
        or baseline is None
        or _number(predicted) >= _number(baseline)
    )
    predicted_is_plausible = (
        predicted is None
        or baseline is None
        or _number(predicted) >= _number(baseline) - 0.15
    )

    if (
        public_rating >= SAFE_PRIMARY_PUBLIC_FLOOR
        and recommendation >= SAFE_PRIMARY_RECOMMENDATION_FLOOR
        and interest >= SAFE_PRIMARY_INTEREST_FLOOR
        and predicted_is_safe
    ):
        tier = 2
        label = "Solide pour toi"
    elif (
        public_rating >= SAFE_SECONDARY_PUBLIC_FLOOR
        and recommendation >= SAFE_SECONDARY_RECOMMENDATION_FLOOR
        and interest >= SAFE_SECONDARY_INTEREST_FLOOR
        and predicted_is_plausible
    ):
        tier = 1
        label = "Plausible pour toi"
    else:
        tier = 0
        label = "Qualité publique seulement"

    score = 100.0 * (
        SAFE_PUBLIC_WEIGHT * public_quality
        + SAFE_RECOMMENDATION_WEIGHT * (recommendation / 100.0)
        + SAFE_INTEREST_WEIGHT * (interest / 100.0)
    )
    return {
        "score": round(score, 3),
        "tier": tier,
        "label": label,
    }


def rank_safe_recommendations(rows: Iterable[dict]) -> list[dict]:
    """Rank reliable public films after a conservative personal guardrail.

    Backtests established that public quality is the strongest satisfaction
    signal among films the user chose to watch.  Production candidates are not
    preselected that way, so compatibility is used as an eligibility tier
    before the public-dominant score is compared.
    """

    ranked = [dict(row) for row in rows]
    for row in ranked:
        evidence = safe_value_evidence(row)
        row["safe_value_score"] = evidence["score"]
        row["safe_eligibility_tier"] = evidence["tier"]
        row["safe_eligibility_label"] = evidence["label"]
    ranked.sort(
        key=lambda row: (
            -int(row["safe_eligibility_tier"]),
            -_number(row["safe_value_score"]),
            -_public_rating(row),
            -_number(row.get("public_rating_reliability")),
            -_number(row.get("vote_count")),
            str(row.get("title") or "").casefold(),
            int(row.get("tmdb_id") or 0),
        )
    )
    for index, row in enumerate(ranked, start=1):
        row["recommended_rank"] = index
        row["safe_rank"] = index
        row["ranking_mode"] = SAFE_LIST_LABEL
        row["recommendation_view"] = "safe"
        row["ranking_exploration"] = 0
    return ranked


def rank_classic_recommendations(rows: Iterable[dict]) -> list[dict]:
    """Use the public-safe order for the independent classic lane."""

    ranked = rank_safe_recommendations(rows)
    for index, row in enumerate(ranked, start=1):
        row["recommended_rank"] = index
        row["classic_rank"] = index
        row["ranking_mode"] = CLASSIC_LIST_LABEL
        row["recommendation_view"] = "classics"
    return ranked


def rank_discovery_recommendations(
    rows: Iterable[dict],
    *,
    excluded_tmdb_ids: set[int] | None = None,
    maximum_probability_gap: float = 8.0,
) -> list[dict]:
    """Rank personal discoveries while keeping the existing diversity guard."""

    excluded = excluded_tmdb_ids or set()
    base = [
        dict(row)
        for row in rows
        if int(row.get("tmdb_id") or 0) not in excluded
    ]
    base.sort(
        key=lambda row: (
            -_number(row.get("discovery_score")),
            -_number(row.get("recommendation_score")),
            -_number(row.get("like_probability")),
            -_number(row.get("confidence")),
            int(row.get("tmdb_id") or 0),
        )
    )
    result: list[dict] = []
    remaining = list(base)
    diversity = float(RANKING_MODES["Découvertes"]["diversity"])
    while remaining:
        strongest_remaining = max(
            _number(
                row.get(
                    "recommendation_score",
                    row.get("like_probability"),
                )
            )
            for row in remaining
        )
        eligible = [
            row
            for row in remaining
            if _number(
                row.get(
                    "recommendation_score",
                    row.get("like_probability"),
                )
            )
            >= strongest_remaining - maximum_probability_gap
        ]

        def utility(row: dict) -> tuple[float, float, float]:
            redundancy = max(
                (
                    candidate_similarity(row, selected)
                    for selected in result[-8:]
                ),
                default=0.0,
            )
            return (
                _number(row.get("discovery_score")) / 100.0
                - diversity * redundancy,
                _number(row.get("recommendation_score")),
                _number(row.get("like_probability")),
            )

        chosen = max(eligible, key=utility)
        result.append(chosen)
        remaining.remove(chosen)

    for index, row in enumerate(result, start=1):
        row["recommended_rank"] = index
        row["discovery_rank"] = index
        row["ranking_mode"] = DISCOVERY_LIST_LABEL
        row["recommendation_view"] = "discovery"
        row["ranking_exploration"] = int(
            RANKING_MODES["Découvertes"]["exploration"]
        )
    return result


def build_recommendation_lists(
    rows: Iterable[dict],
    *,
    safe_exclusion_count: int = SAFE_TOP_EXCLUDED_FROM_DISCOVERIES,
) -> dict[str, list[dict]]:
    """Build recent lists and the independent older-catalogue list."""

    source = [dict(row) for row in rows]
    def is_classic(row: dict) -> bool:
        return (
            row.get("recommendation_lane") == "classics"
            or SOURCE_BACK_CATALOG in row.get("sources", [])
        )

    recent = [row for row in source if not is_classic(row)]
    classic_source = [row for row in source if is_classic(row)]
    safe = rank_safe_recommendations(recent)
    excluded_ids = {
        int(row["tmdb_id"])
        for row in safe[: max(0, safe_exclusion_count)]
        if row.get("tmdb_id") is not None
    }
    discoveries = rank_discovery_recommendations(
        recent,
        excluded_tmdb_ids=excluded_ids,
    )
    return {
        "safe": safe,
        "discovery": discoveries,
        "classics": rank_classic_recommendations(classic_source),
    }


def rerank_recommendations(
    rows: list[dict],
    *,
    mode: str,
    custom_exploration: int | None = None,
    maximum_probability_gap: float = 8.0,
) -> list[dict]:
    if not rows:
        return []
    exploration = exploration_for_mode(mode, custom_exploration)
    diversity = (
        float(RANKING_MODES[mode]["diversity"])
        if custom_exploration is None
        else 0.22 * exploration / 65.0
    )
    saturation_enabled = mode == "Valeurs sûres"
    if exploration == 0:
        def key(row: dict) -> tuple[float, float, float]:
            return (
                float(
                    row.get(
                        "recommendation_score",
                        row.get("like_probability") or 0.0,
                    )
                ),
                float(row.get("interest_score") or 0.0),
                float(row.get("like_probability") or 0.0),
            )
    else:
        def key(row: dict) -> tuple[float, float, float]:
            return (
                float(row.get("rank_score") or 0.0),
                float(row.get("like_probability") or 0.0),
                float(row.get("confidence") or 0.0),
            )
    base = sorted(rows, key=key, reverse=True)
    if diversity <= 0 and not saturation_enabled:
        result = base
    else:
        result: list[dict] = []
        remaining = list(base)
        while remaining:
            safest_remaining = max(
                float(
                    row.get(
                        "recommendation_score",
                        row.get("like_probability") or 0.0,
                    )
                )
                for row in remaining
            )
            eligible = [
                row
                for row in remaining
                if float(
                    row.get(
                        "recommendation_score",
                        row.get("like_probability") or 0.0,
                    )
                )
                >= safest_remaining - maximum_probability_gap
            ]

            def utility(row: dict) -> tuple[float, float, float]:
                redundancy = max(
                    (
                        candidate_similarity(row, selected)
                        for selected in result[-8:]
                    ),
                    default=0.0,
                )
                negative_saturation = 0.0
                if len(result) < 10:
                    for genre in _normalised(
                        row.get("negative_genres", [])
                    ):
                        previous = sum(
                            genre
                            in _normalised(
                                selected.get("negative_genres", [])
                            )
                            for selected in result
                        )
                        if previous >= 2:
                            negative_saturation = max(
                                negative_saturation,
                                0.20 + 0.05 * (previous - 2),
                            )
                return (
                    float(row.get("rank_score") or 0.0) / 100.0
                    - diversity * redundancy
                    - negative_saturation,
                    float(
                        row.get(
                            "recommendation_score",
                            row.get("like_probability") or 0.0,
                        )
                    ),
                    float(row.get("like_probability") or 0.0),
                )

            chosen = max(eligible, key=utility)
            result.append(chosen)
            remaining.remove(chosen)
    for index, row in enumerate(result, start=1):
        row["recommended_rank"] = index
        row["ranking_mode"] = mode
        row["ranking_exploration"] = exploration
    return result

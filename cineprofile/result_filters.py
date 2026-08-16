from __future__ import annotations

from collections.abc import Iterable


def runtime_matches(
    runtime: object,
    runtime_range: tuple[int, int],
) -> bool:
    if runtime in (None, "", 0):
        return runtime_range == (30, 300)
    try:
        minutes = int(runtime)
    except (TypeError, ValueError):
        return False
    return runtime_range[0] <= minutes <= runtime_range[1]


def filter_recommendations(
    recommendations: Iterable[dict],
    *,
    minimum_score: float,
    minimum_interest: float = 0,
    minimum_public_rating: float = 0,
    minimum_imdb_rating: float = 0,
    genres: set[str],
    platforms: set[str],
    languages: set[str],
    runtime_range: tuple[int, int],
    availability: str,
) -> list[dict]:
    visible: list[dict] = []
    for item in recommendations:
        score = item.get("like_probability", item.get("affinity_index", 0))
        if float(score or 0) < minimum_score:
            continue
        if float(item.get("interest_score") or 0) < minimum_interest:
            continue
        if float(
            item.get(
                "bayesian_rating",
                item.get("public_rating_adjusted") or 0,
            )
            or 0
        ) < minimum_public_rating:
            continue
        imdb_rating = item.get("imdb_rating")
        if minimum_imdb_rating > 0 and (
            imdb_rating is None or float(imdb_rating) < minimum_imdb_rating
        ):
            continue
        if genres and not (set(item.get("genres", [])) & genres):
            continue
        item_platforms = {
            platform
            for rows in item.get("providers_ch", {}).values()
            for platform in rows
        }
        if platforms and not (item_platforms & platforms):
            continue
        if languages and item.get("original_language") not in languages:
            continue
        if not runtime_matches(item.get("runtime_minutes"), runtime_range):
            continue
        providers = item.get("providers_ch", {})
        if availability == "Incluse/Gratuite" and not any(
            providers.get(kind) for kind in ("flatrate", "free", "ads")
        ):
            continue
        if availability == "Location/Achat" and not any(
            providers.get(kind) for kind in ("rent", "buy")
        ):
            continue
        if availability == "Disponible en CH" and not providers:
            continue
        visible.append(item)
    return visible


def sort_recommendations(
    recommendations: Iterable[dict],
    *,
    field: str,
    descending: bool,
) -> list[dict]:
    rows = list(recommendations)
    with_value = [item for item in rows if item.get(field) is not None]
    without_value = [item for item in rows if item.get(field) is None]
    with_value.sort(
        key=lambda item: item.get(field),
        reverse=descending,
    )
    return with_value + without_value

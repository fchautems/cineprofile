from __future__ import annotations

import math
from dataclasses import dataclass


TMDB_PRIOR_MEAN = 6.4
TMDB_PRIOR_WEIGHT = 600.0
IMDB_PRIOR_MEAN = 6.8
IMDB_PRIOR_WEIGHT = 1_500.0


def _number(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


@dataclass(frozen=True)
class PublicRating:
    source: str
    raw_rating: float | None
    vote_count: float
    adjusted_rating: float | None
    reliability: float
    prior_mean: float
    prior_weight: float


def public_rating(
    rating: object,
    votes: object,
    *,
    source: str,
) -> PublicRating:
    normalized_source = str(source or "tmdb").casefold()
    if normalized_source == "imdb":
        prior_mean = IMDB_PRIOR_MEAN
        prior_weight = IMDB_PRIOR_WEIGHT
    else:
        normalized_source = "tmdb"
        prior_mean = TMDB_PRIOR_MEAN
        prior_weight = TMDB_PRIOR_WEIGHT

    raw_rating = _number(rating)
    vote_count = max(0.0, _number(votes) or 0.0)
    if (
        raw_rating is None
        or raw_rating <= 0.0
        or raw_rating > 10.0
        or vote_count <= 0.0
    ):
        raw_rating = None
    reliability = (
        vote_count / (vote_count + prior_weight)
        if raw_rating is not None
        else 0.0
    )
    adjusted = (
        reliability * raw_rating + (1.0 - reliability) * prior_mean
        if raw_rating is not None
        else None
    )
    return PublicRating(
        source=normalized_source,
        raw_rating=raw_rating,
        vote_count=vote_count,
        adjusted_rating=adjusted,
        reliability=reliability,
        prior_mean=prior_mean,
        prior_weight=prior_weight,
    )


def best_public_rating(
    *,
    tmdb_rating: object,
    tmdb_votes: object,
    imdb_rating: object = None,
    imdb_votes: object = None,
) -> PublicRating:
    tmdb = public_rating(tmdb_rating, tmdb_votes, source="tmdb")
    imdb = public_rating(imdb_rating, imdb_votes, source="imdb")
    available = [
        rating
        for rating in (tmdb, imdb)
        if rating.raw_rating is not None
    ]
    if not available:
        return tmdb
    # Utiliser la source la mieux étayée. Une note TMDB sur quelques dizaines
    # de votes ne doit pas remplacer une note IMDb fondée sur des milliers de
    # votes uniquement parce que TMDB est présent.
    return max(
        available,
        key=lambda rating: (
            rating.reliability,
            rating.source == "tmdb",
        ),
    )

from __future__ import annotations

import math
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Iterable


INTEREST_MODEL_VERSION = "cineprofile-watch-interest-0.11.0"
INTEREST_ENTITY_TYPE = "interest"

# These defaults describe attraction before watching, not the rating that would
# be given afterwards.  Every factor is visible and editable in Preferences.
INTEREST_FACTOR_DEFINITIONS = {
    "liked_creators": {
        "label": "Réalisateurs et scénaristes réellement appréciés",
        "default": 2,
        "description": "Valorise une signature plusieurs fois vue et appréciée.",
    },
    "familiar_cast": {
        "label": "Acteurs principaux familiers",
        "default": 1,
        "description": "Utilise seulement les premiers rôles déjà bien observés.",
    },
    "liked_predecessor": {
        "label": "Suite directe d’un film apprécié",
        "default": 2,
        "description": "Compare la suite à l’épisode voisin retrouvé dans l’historique.",
    },
    "watchlist_similarity": {
        "label": "Proche d’un film ajouté À voir",
        "default": 1,
        "description": (
            "Utilise les films enregistrés À voir comme un signal d’envie, "
            "sans les considérer comme des films déjà aimés."
        ),
    },
    "dark_comedy": {
        "label": "Comédie noire et satire",
        "default": 2,
        "description": "Distingue la comédie noire d’une comédie générique.",
    },
    "neo_noir": {
        "label": "Néo-noir et mystère criminel",
        "default": 1,
        "description": "Valorise les traitements criminels ou mystérieux précis.",
    },
    "western": {
        "label": "Western",
        "default": 1,
        "description": "Ajoute un attrait modéré au western.",
    },
    "thriller": {
        "label": "Thriller",
        "default": 1,
        "description": "Reste volontairement faible : le traitement compte davantage.",
    },
    "biography": {
        "label": "Biographie et biopic",
        "default": -1,
        "description": "Réduit l’envie sans prétendre que le film serait mal noté.",
    },
    "historical": {
        "label": "Sujet principalement historique",
        "default": -1,
        "description": "Pénalise surtout l’histoire sans autre accroche personnelle.",
    },
    "music_biopic": {
        "label": "Biographie musicale ou film centré sur un chanteur",
        "default": -2,
        "description": "Frein renforcé lorsqu’un biopic est aussi musical.",
    },
    "sports_fighting": {
        "label": "Sport, boxe, combat ou UFC",
        "default": -2,
        "description": "Réduit fortement les sujets sportifs ou de combat.",
    },
    "animation": {
        "label": "Animation",
        "default": -1,
        "description": "Frein modéré, compensable par un film précédent très apprécié.",
    },
    "sequel": {
        "label": "Usure des suites",
        "default": -1,
        "description": "La pénalité augmente pour un troisième épisode ou davantage.",
    },
    "unfamiliar_language": {
        "label": "Langue ou cinéma peu familier",
        "default": -1,
        "description": "Petit frein seulement, jamais une exclusion automatique.",
    },
    "unknown_team": {
        "label": "Équipe principale sans repère connu",
        "default": -1,
        "description": "Évite qu’un genre générique suffise à créer l’envie.",
    },
}

INTEREST_ADJUSTMENT_LABELS = {
    -2: "Réduire fortement",
    -1: "Réduire",
    0: "Neutre",
    1: "Favoriser",
    2: "Favoriser fortement",
}
INTEREST_LABEL_ADJUSTMENTS = {
    label: value for value, label in INTEREST_ADJUSTMENT_LABELS.items()
}


def _normalise(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def interest_adjustments(
    preferences: dict[tuple[str, str], dict],
) -> dict[str, int]:
    values = {
        key: int(configuration["default"])
        for key, configuration in INTEREST_FACTOR_DEFINITIONS.items()
    }
    for key in values:
        row = preferences.get((INTEREST_ENTITY_TYPE, key))
        if row is not None:
            values[key] = max(-2, min(2, int(row["adjustment"])))
    return values


def _keyword_names(candidate: dict) -> list[str]:
    payload = candidate.get("keywords", {})
    rows = payload.get("keywords", payload.get("results", []))
    return [
        _normalise(row.get("name"))
        for row in rows
        if isinstance(row, dict) and row.get("name")
    ]


def _genre_names(candidate: dict) -> set[str]:
    return {
        _normalise(row.get("name"))
        for row in candidate.get("genres", [])
        if isinstance(row, dict) and row.get("name")
    }


def _has_pattern(values: Iterable[str], patterns: set[str]) -> bool:
    return any(
        pattern == value or pattern in value
        for value in values
        for pattern in patterns
    )


def _candidate_text(candidate: dict, keywords: list[str]) -> str:
    return " ".join(
        [
            _normalise(candidate.get("title") or candidate.get("name")),
            _normalise(candidate.get("tagline")),
            _normalise(candidate.get("overview")),
            *keywords,
        ]
    )


def _title_root(value: object) -> str:
    text = _normalise(value)
    text = re.sub(
        r"\b(?:part|partie|chapter|chapitre|episode|film)\s*[ivx0-9]+\b",
        " ",
        text,
    )
    text = re.sub(r"\b(?:ii|iii|iv|v|vi|vii|viii|ix|x|[2-9])\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _installment_number(title: object) -> int | None:
    text = _normalise(title)
    numbers = [
        int(value)
        for value in re.findall(
            r"\b(?:part|partie|chapter|chapitre|episode)?\s*([2-9])\b",
            text,
        )
    ]
    roman = {
        "ii": 2,
        "iii": 3,
        "iv": 4,
        "v": 5,
        "vi": 6,
    }
    numbers.extend(
        roman[value]
        for value in re.findall(r"\b(ii|iii|iv|v|vi)\b", text)
        if value in roman
    )
    return max(numbers) if numbers else None


def _predecessor(
    candidate: dict,
    semantic_neighbors: list[dict],
    *,
    sequel_detected: bool,
) -> dict | None:
    candidate_root = _title_root(
        candidate.get("title") or candidate.get("name")
    )
    best: tuple[float, dict] | None = None
    for neighbor in semantic_neighbors:
        rating = neighbor.get("rating")
        if rating is None:
            continue
        neighbor_root = _title_root(neighbor.get("title"))
        title_similarity = SequenceMatcher(
            None,
            candidate_root,
            neighbor_root,
        ).ratio()
        semantic_similarity = float(neighbor.get("similarity") or 0.0) / 100.0
        plausible = (
            title_similarity >= 0.72
            or (
                sequel_detected
                and semantic_similarity >= 0.74
                and neighbor is semantic_neighbors[0]
            )
        )
        if not plausible:
            continue
        strength = max(title_similarity, semantic_similarity)
        if best is None or strength > best[0]:
            best = (strength, neighbor)
    return best[1] if best else None


def _profile_language_row(profile: dict, language: str) -> dict | None:
    for row in profile.get("dimensions", {}).get("languages", []):
        if _normalise(row.get("name")) == _normalise(language):
            return row
    return None


def _explicit_interest_impacts(
    entities: dict[str, list[tuple[str, str]]],
    preferences: dict[tuple[str, str], dict],
) -> list[dict]:
    impacts: list[dict] = []
    weights = {
        "directors": 10.0,
        "writers": 8.0,
        "actors": 5.0,
        "keywords": 5.0,
        "genres": 3.0,
    }
    for dimension, rows in entities.items():
        weight = weights.get(dimension)
        if weight is None:
            continue
        for entity_id, name in rows:
            preference = preferences.get((dimension, entity_id))
            if preference is None:
                continue
            adjustment = int(preference["adjustment"])
            if adjustment:
                impacts.append(
                    {
                        "factor": "explicit_preference",
                        "label": f"Réglage personnel : {name}",
                        "impact": weight * adjustment,
                    }
                )
    return impacts


def _add_factor(
    impacts: list[dict],
    adjustments: dict[str, int],
    factor: str,
    *,
    strength: float = 1.0,
    unit: float,
    detail: str | None = None,
) -> None:
    adjustment = int(adjustments.get(factor, 0))
    if adjustment == 0 or strength <= 0:
        return
    label = str(INTEREST_FACTOR_DEFINITIONS[factor]["label"])
    if detail:
        label += f" : {detail}"
    impacts.append(
        {
            "factor": factor,
            "label": label,
            "impact": round(unit * adjustment * strength, 2),
        }
    )


def satisfaction_lift_index(
    like_probability: float,
    base_like_rate: float,
) -> float:
    probability = max(0.01, min(0.99, float(like_probability)))
    baseline = max(0.01, min(0.99, float(base_like_rate)))
    value = 50.0 + 40.0 * math.log2(probability / baseline)
    return round(max(5.0, min(95.0, value)), 1)


def combined_recommendation_score(
    interest_score: float,
    satisfaction_index: float,
) -> float:
    interest = max(1.0, min(100.0, float(interest_score)))
    satisfaction = max(1.0, min(100.0, float(satisfaction_index)))
    # The harmonic mean prevents an excellent predicted rating from masking a
    # film that creates little desire, and conversely.
    value = 1.0 / (0.60 / interest + 0.40 / satisfaction)
    return round(max(1.0, min(99.0, value)), 1)


def score_watch_interest(
    candidate: dict,
    *,
    profile: dict,
    entities: dict[str, list[tuple[str, str]]],
    matched_details: list[dict],
    semantic_neighbors: list[dict],
    preferences: dict[tuple[str, str], dict],
    like_probability: float,
    base_like_rate: float,
) -> dict:
    adjustments = interest_adjustments(preferences)
    impacts: list[dict] = _explicit_interest_impacts(entities, preferences)
    genres = _genre_names(candidate)
    keywords = _keyword_names(candidate)
    text = _candidate_text(candidate, keywords)

    creator_rows = [
        row
        for row in matched_details
        if row.get("dimension") in {"directors", "writers"}
        and int(row.get("seen") or 0) >= 5
        and float(row.get("affinity") or 0.0) >= 0.12
        and float(row.get("average_rating") or 0.0)
        >= float(profile["summary"]["average_rating"]) + 0.25
    ]
    if creator_rows:
        strongest = max(
            creator_rows,
            key=lambda row: (
                float(row.get("affinity") or 0.0),
                int(row.get("seen") or 0),
            ),
        )
        strength = min(
            1.25,
            0.55
            + 0.06 * int(strongest.get("seen") or 0)
            + 0.30 * float(strongest.get("affinity") or 0.0),
        )
        _add_factor(
            impacts,
            adjustments,
            "liked_creators",
            strength=strength,
            unit=6.0,
            detail=str(strongest["name"]),
        )

    actor_rows = [
        row
        for row in matched_details
        if row.get("dimension") == "actors"
        and int(row.get("seen") or 0) >= 7
    ]
    if actor_rows:
        names = [row["name"] for row in actor_rows[:2]]
        familiarity = min(
            1.5,
            sum(min(1.0, int(row.get("seen") or 0) / 12.0) for row in actor_rows[:2]),
        )
        _add_factor(
            impacts,
            adjustments,
            "familiar_cast",
            strength=familiarity,
            unit=4.0,
            detail=", ".join(names),
        )

    dark_comedy = _has_pattern(
        keywords,
        {"dark comedy", "black comedy", "satire", "satirical"},
    ) or any(
        phrase in text
        for phrase in ("comedie noire", "humour noir", "dark comedy")
    )
    if dark_comedy:
        _add_factor(impacts, adjustments, "dark_comedy", unit=6.0)

    neo_noir = _has_pattern(
        keywords,
        {
            "neo noir",
            "film noir",
            "whodunit",
            "murder mystery",
            "detective",
        },
    )
    if neo_noir:
        _add_factor(impacts, adjustments, "neo_noir", unit=5.0)
    if "western" in genres:
        _add_factor(impacts, adjustments, "western", unit=5.0)
    if "thriller" in genres:
        _add_factor(impacts, adjustments, "thriller", unit=2.0)

    biography = _has_pattern(
        keywords,
        {"biography", "biopic", "historical figure"},
    ) or any(
        phrase in text
        for phrase in ("biographie", "biopic", "vie et heritage")
    )
    if biography:
        _add_factor(impacts, adjustments, "biography", unit=7.0)

    historical = bool({"histoire", "history"} & genres) or _has_pattern(
        keywords,
        {
            "historical fiction",
            "historical drama",
            "period drama",
            "historical figure",
            "19th century",
            "18th century",
            "17th century",
            "16th century",
            "medieval",
        },
    )
    if historical:
        _add_factor(
            impacts,
            adjustments,
            "historical",
            strength=0.70 if (dark_comedy or neo_noir) else 1.0,
            unit=7.0,
        )

    musical = bool({"musique", "music"} & genres) or _has_pattern(
        keywords,
        {"musician", "singer", "concert", "music industry"},
    )
    if musical and biography:
        _add_factor(impacts, adjustments, "music_biopic", unit=6.0)

    sport_fighting = _has_pattern(
        keywords,
        {
            "sport",
            "sports",
            "boxing",
            "boxer",
            "mixed martial arts",
            "martial arts",
            "ufc",
            "wrestling",
            "fighter",
            "fighting",
            "combat sport",
        },
    ) or any(
        phrase in text
        for phrase in (
            "arts martiaux",
            "combat sportif",
            "ultimate fighting",
            "champion de boxe",
        )
    )
    if sport_fighting:
        _add_factor(impacts, adjustments, "sports_fighting", unit=8.0)

    if "animation" in genres:
        _add_factor(impacts, adjustments, "animation", unit=8.0)

    sequel_detected = _has_pattern(
        keywords,
        {"sequel"},
    ) or _installment_number(candidate.get("title") or candidate.get("name")) is not None
    installment = _installment_number(
        candidate.get("title") or candidate.get("name")
    )
    if sequel_detected:
        sequel_strength = (
            1.75 if installment and installment >= 3 else 1.0
        )
        _add_factor(
            impacts,
            adjustments,
            "sequel",
            strength=sequel_strength,
            unit=4.0,
            detail=(f"épisode {installment}" if installment else None),
        )
    predecessor = _predecessor(
        candidate,
        semantic_neighbors,
        sequel_detected=sequel_detected,
    )
    if predecessor is not None:
        rating = float(predecessor["rating"])
        if rating >= 8.0:
            _add_factor(
                impacts,
                adjustments,
                "liked_predecessor",
                strength=min(1.25, 0.75 + 0.125 * (rating - 6.0)),
                unit=7.0,
                detail=f"{predecessor['title']} ({rating:g}/10)",
            )
        elif rating <= 6.0:
            impacts.append(
                {
                    "factor": "disliked_predecessor",
                    "label": (
                        "Épisode voisin moins aimé : "
                        f"{predecessor['title']} ({rating:g}/10)"
                    ),
                    "impact": round(-6.0 - 2.0 * (6.0 - rating), 2),
                }
            )

    watchlist_neighbors = [
        row
        for row in semantic_neighbors
        if row.get("feedback") == "À voir"
    ]
    if watchlist_neighbors:
        closest_watchlist = max(
            watchlist_neighbors,
            key=lambda row: float(row.get("similarity") or 0.0),
        )
        similarity = max(
            0.0,
            min(
                1.0,
                float(closest_watchlist.get("similarity") or 0.0) / 100.0,
            ),
        )
        if similarity >= 0.30:
            _add_factor(
                impacts,
                adjustments,
                "watchlist_similarity",
                strength=max(
                    0.15,
                    min(1.0, (similarity - 0.25) / 0.45),
                ),
                unit=6.0,
                detail=str(closest_watchlist.get("title") or ""),
            )

    language = str(candidate.get("original_language") or "")
    language_row = _profile_language_row(profile, language)
    unfamiliar_language = (
        bool(language)
        and language not in {"fr", "en"}
        and (
            language_row is None
            or int(language_row.get("seen") or 0) < 8
            or float(language_row.get("affinity") or 0.0) < -0.05
        )
    )
    if unfamiliar_language and not creator_rows and not actor_rows:
        _add_factor(
            impacts,
            adjustments,
            "unfamiliar_language",
            unit=4.0,
            detail=language,
        )

    if not creator_rows and not actor_rows:
        _add_factor(impacts, adjustments, "unknown_team", unit=4.0)

    raw_score = 50.0 + sum(float(row["impact"]) for row in impacts)
    positive_hooks = [
        row for row in impacts if float(row["impact"]) >= 4.0
    ]
    if not positive_hooks:
        raw_score = min(raw_score, 55.0)
    elif len(positive_hooks) == 1:
        raw_score = min(raw_score, 68.0)
    interest_score = round(max(5.0, min(95.0, raw_score)), 1)
    positive = sorted(
        [row for row in impacts if float(row["impact"]) > 0],
        key=lambda row: float(row["impact"]),
        reverse=True,
    )
    negative = sorted(
        [row for row in impacts if float(row["impact"]) < 0],
        key=lambda row: float(row["impact"]),
    )
    evidence_count = len(positive) + len(negative)
    interest_confidence = round(
        100.0
        * min(
            0.95,
            0.35
            + 0.08 * min(5, evidence_count)
            + 0.12 * bool(creator_rows or actor_rows or predecessor),
        ),
        1,
    )
    interest_label = (
        "Élevée"
        if interest_score >= 70.0
        else "Moyenne"
        if interest_score >= 52.0
        else "Faible"
    )
    satisfaction_index = satisfaction_lift_index(
        like_probability,
        base_like_rate,
    )
    recommendation_score = combined_recommendation_score(
        interest_score,
        satisfaction_index,
    )
    return {
        "model_version": INTEREST_MODEL_VERSION,
        "interest_score": interest_score,
        "interest_label": interest_label,
        "interest_confidence": interest_confidence,
        "positive_reasons": positive[:6],
        "reservations": negative[:6],
        "satisfaction_lift_index": satisfaction_index,
        "recommendation_score": recommendation_score,
        "base_like_rate": round(100.0 * float(base_like_rate), 1),
        "like_probability_lift_points": round(
            100.0 * (float(like_probability) - float(base_like_rate)),
            1,
        ),
        "like_probability_lift_ratio": round(
            float(like_probability) / max(0.01, float(base_like_rate)),
            2,
        ),
    }

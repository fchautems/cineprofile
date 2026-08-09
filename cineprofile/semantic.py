from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .db import connect, initialize, transaction
from .diagnostics import configure_logging
from .media_types import is_series_type


SEMANTIC_MODEL = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
EmbeddingProgressCallback = Callable[[int, int, str], None]


def embedding_execution_providers() -> list[str]:
    """Resolve the explicitly requested ONNX execution backend.

    CPU remains the safe default for CineProfile.  The semantic laboratory can
    opt into CUDA through its dedicated launcher and isolated environment.
    """

    requested = os.getenv(
        "CINEPROFILE_SEMANTIC_DEVICE",
        "cpu",
    ).strip().casefold()
    if requested in {"", "cpu"}:
        return ["CPUExecutionProvider"]
    if requested != "cuda":
        raise RuntimeError(
            "CINEPROFILE_SEMANTIC_DEVICE doit valoir 'cpu' ou 'cuda'."
        )

    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "ONNX Runtime GPU est absent de l’environnement GPU."
        ) from exc

    preload = getattr(ort, "preload_dlls", None)
    if callable(preload):
        # The isolated GPU environment installs CUDA and cuDNN as Python
        # packages.  Asking ORT to preload from site-packages avoids requiring
        # a system-wide CUDA Toolkit installation on Windows.
        preload(directory="")
    available = list(ort.get_available_providers())
    if "CUDAExecutionProvider" not in available:
        raise RuntimeError(
            "La carte NVIDIA est détectée, mais CUDAExecutionProvider est "
            "indisponible. Mets à jour le pilote NVIDIA puis relance le "
            "laboratoire GPU."
        )
    return ["CUDAExecutionProvider", "CPUExecutionProvider"]


def _document_text(item: dict) -> str:
    def names(value: object) -> list[str]:
        if isinstance(value, dict):
            value = value.get("keywords", value.get("results", []))
        if not isinstance(value, list):
            return []
        result: list[str] = []
        for entry in value:
            name = entry.get("name") if isinstance(entry, dict) else entry
            if name and str(name).strip():
                result.append(str(name).strip())
        return result

    parts: list[str] = []
    overview = str(item.get("overview") or "").strip()
    tagline = str(item.get("tagline") or "").strip()
    genres = names(item.get("genres"))
    keywords = names(item.get("keywords"))
    if overview:
        parts.append("Résumé : " + overview)
    if tagline:
        parts.append("Accroche : " + tagline)
    if genres:
        parts.append("Genres : " + ", ".join(genres))
    if keywords:
        parts.append("Thèmes : " + ", ".join(keywords))
    return " ".join(parts)


def _normalise_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _wanted_hashes(
    documents: list[dict],
) -> dict[tuple[str, str], str]:
    return {
        (item["kind"], str(item["id"])): hashlib.sha256(
            item["text"].encode("utf-8")
        ).hexdigest()
        for item in documents
    }


def _read_cached_embeddings(
    database: str | Path | None,
    wanted: dict[tuple[str, str], str],
    *,
    model_name: str,
) -> dict[tuple[str, str], np.ndarray]:
    cached: dict[tuple[str, str], np.ndarray] = {}
    with connect(database) as connection:
        rows = connection.execute(
            """
            SELECT item_kind, item_id, text_hash, dimensions, vector_blob
            FROM text_embeddings WHERE model_name=?
            """,
            (model_name,),
        ).fetchall()
    for row in rows:
        key = (str(row["item_kind"]), str(row["item_id"]))
        if wanted.get(key) != str(row["text_hash"]):
            continue
        vector = np.frombuffer(row["vector_blob"], dtype=np.float32)
        if len(vector) == int(row["dimensions"]):
            cached[key] = vector
    return cached


def embedding_cache_coverage(
    database: str | Path | None,
    documents: list[dict],
    *,
    model_name: str = SEMANTIC_MODEL,
) -> dict[str, int]:
    """Count valid cached vectors without creating or downloading anything."""

    if not documents:
        return {"total": 0, "cached": 0, "missing": 0}
    initialize(database)
    wanted = _wanted_hashes(documents)
    cached = _read_cached_embeddings(
        database,
        wanted,
        model_name=model_name,
    )
    return {
        "total": len(documents),
        "cached": len(cached),
        "missing": max(0, len(documents) - len(cached)),
    }


def _load_or_create_embeddings(
    database: str | Path | None,
    documents: list[dict],
    *,
    model_name: str,
    cache_directory: str | Path | None = None,
    batch_size: int = 64,
    on_progress: EmbeddingProgressCallback | None = None,
) -> np.ndarray:
    from fastembed import TextEmbedding

    wanted = _wanted_hashes(documents)
    cached = _read_cached_embeddings(
        database,
        wanted,
        model_name=model_name,
    )

    missing = [
        item
        for item in documents
        if (item["kind"], str(item["id"])) not in cached
    ]
    if missing:
        target = Path(database) if database else Path("data/cineprofile.db")
        model_cache = (
            Path(cache_directory)
            if cache_directory is not None
            else target.parent / "models"
        )
        model_cache.mkdir(parents=True, exist_ok=True)
        missing_count = len(missing)
        if on_progress:
            on_progress(
                0,
                missing_count,
                f"Calcul local de {missing_count} vecteurs sémantiques",
            )
        model = TextEmbedding(
            model_name=model_name,
            cache_dir=str(model_cache),
            threads=max(1, min(6, __import__("os").cpu_count() or 2)),
            providers=embedding_execution_providers(),
        )
        effective_batch_size = max(1, int(batch_size))
        vectors: list[np.ndarray] = []
        progress_step = max(1, missing_count // 20)
        for position, vector in enumerate(
            model.embed(
                [item["text"] for item in missing],
                batch_size=effective_batch_size,
            ),
            start=1,
        ):
            vectors.append(np.asarray(vector, dtype=np.float32))
            if on_progress and (
                position == 1
                or position == missing_count
                or position % progress_step == 0
            ):
                on_progress(
                    position,
                    missing_count,
                    (
                        "Calcul sémantique local "
                        f"· {position}/{missing_count}"
                    ),
                )
        if len(vectors) != missing_count:
            raise RuntimeError(
                "Le modèle sémantique n’a pas produit tous les vecteurs "
                f"attendus ({len(vectors)}/{missing_count})."
            )
        now = datetime.now(UTC).isoformat()
        with transaction(database) as connection:
            connection.executemany(
                """
                INSERT INTO text_embeddings(
                  item_kind, item_id, model_name, text_hash, dimensions,
                  vector_blob, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(item_kind, item_id, model_name) DO UPDATE SET
                  text_hash=excluded.text_hash,
                  dimensions=excluded.dimensions,
                  vector_blob=excluded.vector_blob,
                  updated_at=excluded.updated_at
                """,
                [
                    (
                        item["kind"],
                        str(item["id"]),
                        model_name,
                        wanted[(item["kind"], str(item["id"]))],
                        int(len(vector)),
                        vector.tobytes(),
                        now,
                    )
                    for item, vector in zip(missing, vectors, strict=True)
                ],
            )
        for item, vector in zip(missing, vectors, strict=True):
            cached[(item["kind"], str(item["id"]))] = vector
    elif on_progress:
        on_progress(0, 0, "Cache sémantique déjà complet")

    return _normalise_rows(
        np.vstack(
            [cached[(item["kind"], str(item["id"]))] for item in documents]
        )
    )


def cached_text_embeddings(
    database: str | Path | None,
    documents: list[dict],
    *,
    model_name: str = SEMANTIC_MODEL,
    cache_directory: str | Path | None = None,
    batch_size: int = 64,
    on_progress: EmbeddingProgressCallback | None = None,
) -> np.ndarray:
    """Return normalized local embeddings with a persistent, text-hash cache.

    The caller supplies only an item kind, an id and the text.  Ratings and
    other private preference data are never sent to the embedding model.  The
    model runs locally and unchanged text is encoded only once.
    """

    if not documents:
        return np.empty((0, 0), dtype=np.float32)
    initialize(database)
    return _load_or_create_embeddings(
        database,
        documents,
        model_name=model_name,
        cache_directory=cache_directory,
        batch_size=batch_size,
        on_progress=on_progress,
    )


def _evidence_from_similarities(
    similarities: np.ndarray,
    references: list[dict],
    candidates: list[dict],
    *,
    baseline: float = 7.0,
    base_like_rate: float = 0.20,
    threshold: float,
    engine: str,
    error: str | None = None,
) -> dict[int, dict]:
    evidence: dict[int, dict] = {}
    for candidate_index, candidate in enumerate(candidates):
        ordered = np.argsort(similarities[candidate_index])[::-1]
        selected = [
            int(index)
            for index in ordered[:24]
            if float(similarities[candidate_index, index]) >= threshold
        ][:16]
        if not selected:
            evidence[int(candidate["id"])] = {
                "score": None,
                "raw_like_probability": None,
                "predicted_rating": None,
                "base_like_rate": base_like_rate,
                "confidence": 0.0,
                "similarity": 0.0,
                "positive_similarity": 0.0,
                "negative_similarity": 0.0,
                "neighbors": [],
                "engine": engine,
                "error": error,
            }
            continue

        similarity_values = np.array(
            [float(similarities[candidate_index, index]) for index in selected],
            dtype=float,
        )
        weights = np.square(np.maximum(0.03, similarity_values - threshold + 0.08))
        satisfaction_positions = [
            position
            for position, reference_index in enumerate(selected)
            if references[reference_index].get("reference_role")
            != "interest"
        ]
        satisfaction_indexes = [
            selected[position] for position in satisfaction_positions
        ]
        satisfaction_weights = weights[satisfaction_positions]
        satisfaction_similarities = similarity_values[
            satisfaction_positions
        ]
        values = np.array(
            [
                float(references[index]["residual"])
                for index in satisfaction_indexes
            ],
            dtype=float,
        )
        if not len(values):
            values = np.asarray([0.0], dtype=float)
            satisfaction_weights = np.asarray([1.0], dtype=float)
            satisfaction_similarities = np.asarray([0.0], dtype=float)
        signal = float(np.average(values, weights=satisfaction_weights))
        liked = values >= 0.5
        disliked = values <= -0.5
        positive_similarity = (
            float(np.max(satisfaction_similarities[liked]))
            if np.any(liked)
            else 0.0
        )
        negative_similarity = (
            float(np.max(satisfaction_similarities[disliked]))
            if np.any(disliked)
            else 0.0
        )
        # Les ressemblances avec les films mal notés ou explicitement refusés
        # comptent davantage que la simple nouveauté. Cela évite qu'un voisin
        # négatif très proche soit noyé dans plusieurs ressemblances moyennes.
        negative_margin = max(
            0.0,
            negative_similarity - positive_similarity,
        )
        signal -= 2.0 * negative_margin
        agreement = 1.0 / (1.0 + float(np.std(values)) / 2.0)
        confidence = (
            min(1.0, len(satisfaction_indexes) / 8.0) * agreement
            if satisfaction_indexes
            else 0.0
        )
        if engine == "semantic":
            confidence *= min(
                1.0,
                max(
                    0.15,
                    (
                        float(np.mean(satisfaction_similarities)) - 0.25
                    )
                    / 0.35,
                ),
            )
        else:
            confidence *= min(
                0.55,
                max(
                    0.10,
                    float(np.mean(satisfaction_similarities)) / 0.22,
                ),
            )
        like_values = np.asarray(
            [
                1.0
                if references[index].get("rating") is not None
                and float(references[index]["rating"]) >= 8.0
                else 0.0
                for index in satisfaction_indexes
            ],
            dtype=float,
        )
        raw_like_probability = (
            float(
                np.average(
                    like_values,
                    weights=satisfaction_weights,
                )
            )
            if len(like_values)
            else base_like_rate
        )
        probability = base_like_rate + confidence * (
            raw_like_probability - base_like_rate
        )
        probability -= min(0.25, 0.35 * negative_margin)
        probability = max(0.01, min(0.99, probability))
        rated_indexes = [
            position
            for position, reference_index in enumerate(satisfaction_indexes)
            if references[reference_index].get("rating") is not None
        ]
        if rated_indexes:
            neighbor_rating = float(
                np.average(
                    [
                        float(
                            references[
                                satisfaction_indexes[position]
                            ]["rating"]
                        )
                        for position in rated_indexes
                    ],
                    weights=satisfaction_weights[rated_indexes],
                )
            )
            predicted_rating = baseline + confidence * (
                neighbor_rating - baseline
            )
        else:
            predicted_rating = baseline + min(1.5, max(-1.5, signal))

        neighbors = [
            {
                "title": references[index]["title"],
                "rating": references[index].get("rating"),
                "feedback": references[index].get("feedback"),
                "sentiment": (
                    "aimé"
                    if float(references[index]["residual"]) >= 0.5
                    else "moins aimé"
                    if float(references[index]["residual"]) <= -0.5
                    else "neutre"
                ),
                "similarity": round(
                    100 * float(similarities[candidate_index, index]),
                    1,
                ),
            }
            for index in selected[:5]
        ]
        evidence[int(candidate["id"])] = {
            "score": float(probability),
            "raw_like_probability": raw_like_probability,
            "predicted_rating": float(
                max(1.0, min(10.0, predicted_rating))
            ),
            "base_like_rate": base_like_rate,
            "confidence": float(confidence),
            "similarity": float(similarity_values[0]),
            "positive_similarity": positive_similarity,
            "negative_similarity": negative_similarity,
            "neighbors": neighbors,
            "engine": engine,
            "error": error,
        }
    return evidence


def _lexical_fallback(
    references: list[dict],
    candidates: list[dict],
    *,
    baseline: float,
    base_like_rate: float,
    error: str | None,
) -> dict[int, dict]:
    reference_texts = [item["text"] for item in references]
    candidate_texts = [_document_text(item) for item in candidates]
    if len(reference_texts) < 3 or not any(candidate_texts):
        return {
            int(item["id"]): {
                "score": None,
                "raw_like_probability": None,
                "predicted_rating": None,
                "base_like_rate": base_like_rate,
                "confidence": 0.0,
                "similarity": 0.0,
                "positive_similarity": 0.0,
                "negative_similarity": 0.0,
                "neighbors": [],
                "engine": "lexical",
                "error": error,
            }
            for item in candidates
        }
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_features=16000,
        strip_accents="unicode",
    )
    matrix = vectorizer.fit_transform(reference_texts + candidate_texts)
    similarities = cosine_similarity(
        matrix[len(reference_texts) :],
        matrix[: len(reference_texts)],
    )
    return _evidence_from_similarities(
        similarities,
        references,
        candidates,
        baseline=baseline,
        base_like_rate=base_like_rate,
        threshold=0.04,
        engine="lexical",
        error=error,
    )


def semantic_evidence(
    database: str | Path | None,
    candidates: list[dict],
    baseline: float,
    *,
    enabled: bool = True,
) -> dict[int, dict]:
    initialize(database)
    with connect(database) as connection:
        rows = connection.execute(
            """
            SELECT imdb_id, title, title_type, overview, tagline, user_rating
                 , (
                     SELECT GROUP_CONCAT(g.name, ', ')
                     FROM title_genres tg
                     JOIN genres g ON g.tmdb_id=tg.genre_id
                     WHERE tg.imdb_id=titles.imdb_id
                   ) AS genre_names
                 , (
                     SELECT GROUP_CONCAT(k.name, ', ')
                     FROM title_keywords tk
                     JOIN keywords k ON k.tmdb_id=tk.keyword_id
                     WHERE tk.imdb_id=titles.imdb_id
                   ) AS keyword_names
            FROM titles
            WHERE metadata_status='done'
              AND overview IS NOT NULL
              AND TRIM(overview)<>''
            """
        ).fetchall()
        feedback_rows = connection.execute(
            """
            SELECT tmdb_id, title, action, payload_json
            FROM recommendation_feedback
            WHERE action IN ('not_interested', 'watchlist')
              AND payload_json IS NOT NULL
            """
        ).fetchall()

    references = [
        {
            "kind": "title",
            "id": row["imdb_id"],
            "title": row["title"],
            "text": _document_text(
                {
                    "overview": row["overview"],
                    "tagline": row["tagline"],
                    "genres": (
                        str(row["genre_names"]).split(", ")
                        if row["genre_names"]
                        else []
                    ),
                    "keywords": (
                        str(row["keyword_names"]).split(", ")
                        if row["keyword_names"]
                        else []
                    ),
                }
            ),
            "rating": float(row["user_rating"]),
            "residual": float(row["user_rating"]) - baseline,
        }
        for row in rows
        if not is_series_type(row["title_type"])
    ]
    for row in feedback_rows:
        try:
            payload = json.loads(row["payload_json"])
        except (TypeError, json.JSONDecodeError):
            continue
        text = _document_text(payload)
        if text:
            is_watchlist = str(row["action"]) == "watchlist"
            references.append(
                {
                    "kind": "feedback",
                    "id": str(row["tmdb_id"]),
                    "title": row["title"],
                    "text": text,
                    "feedback": "À voir" if is_watchlist else "Pas intéressé",
                    "reference_role": (
                        "interest" if is_watchlist else "satisfaction"
                    ),
                    "residual": 0.0 if is_watchlist else -3.0,
                }
            )

    rated_references = [
        row for row in references if row.get("rating") is not None
    ]
    base_like_rate = (
        sum(float(row["rating"]) >= 8.0 for row in rated_references)
        / len(rated_references)
        if rated_references
        else 0.20
    )
    usable_candidates = [
        candidate for candidate in candidates if _document_text(candidate)
    ]
    if len(references) < 3 or not usable_candidates:
        return _lexical_fallback(
            references,
            candidates,
            baseline=baseline,
            base_like_rate=base_like_rate,
            error=None,
        )
    if not enabled:
        return _lexical_fallback(
            references,
            candidates,
            baseline=baseline,
            base_like_rate=base_like_rate,
            error=None,
        )

    documents = references + [
        {
            "kind": "candidate",
            "id": str(candidate["id"]),
            "text": _document_text(candidate),
        }
        for candidate in usable_candidates
    ]
    try:
        matrix = _load_or_create_embeddings(
            database,
            documents,
            model_name=SEMANTIC_MODEL,
        )
        reference_matrix = matrix[: len(references)]
        candidate_matrix = matrix[len(references) :]
        similarities = candidate_matrix @ reference_matrix.T
        result = _evidence_from_similarities(
            similarities,
            references,
            usable_candidates,
            baseline=baseline,
            base_like_rate=base_like_rate,
            threshold=0.30,
            engine="semantic",
        )
        for candidate in candidates:
            result.setdefault(
                int(candidate["id"]),
                {
                    "score": None,
                    "raw_like_probability": None,
                    "predicted_rating": None,
                    "base_like_rate": base_like_rate,
                    "confidence": 0.0,
                    "similarity": 0.0,
                    "positive_similarity": 0.0,
                    "negative_similarity": 0.0,
                    "neighbors": [],
                    "engine": "semantic",
                    "error": None,
                },
            )
        return result
    except Exception as exc:
        configure_logging(database).warning(
            "semantic_fallback | engine=%s | error=%s",
            SEMANTIC_MODEL,
            str(exc)[:300],
        )
        return _lexical_fallback(
            references,
            candidates,
            baseline=baseline,
            base_like_rate=base_like_rate,
            error=str(exc)[:300],
        )

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from . import hybrid_model as hm
from .db import connect, initialize
from .media_types import is_series_type
from .public_rating import best_public_rating
from .semantic import cached_text_embeddings


@dataclass(frozen=True)
class SemanticModelSpec:
    key: str
    label: str
    model_name: str
    dimensions: int
    download_gib: float
    batch_size: int
    text_prefix: str = ""
    custom_pooling: str | None = None
    model_file: str = "onnx/model.onnx"
    additional_files: tuple[str, ...] = ()
    cache_repositories: tuple[str, ...] = ()

    @property
    def engine(self) -> str:
        return f"semantic_{self.key}_structured_dense_v1"


SEMANTIC_MODELS = (
    SemanticModelSpec(
        key="minilm",
        label="MiniLM multilingue",
        model_name=(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        ),
        dimensions=384,
        download_gib=0.22,
        batch_size=64,
        cache_repositories=(
            "qdrant/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q",
        ),
    ),
    SemanticModelSpec(
        key="e5_large",
        label="Multilingual E5 Large",
        model_name="intfloat/multilingual-e5-large",
        dimensions=1024,
        download_gib=2.24,
        batch_size=8,
        text_prefix="passage: ",
        cache_repositories=("qdrant/multilingual-e5-large-onnx",),
    ),
    SemanticModelSpec(
        key="bge_m3",
        label="BGE-M3",
        model_name="BAAI/bge-m3",
        dimensions=1024,
        download_gib=2.27,
        batch_size=8,
        custom_pooling="CLS",
        additional_files=("onnx/model.onnx_data",),
        cache_repositories=("BAAI/bge-m3",),
    ),
)
MODEL_BY_KEY = {row.key: row for row in SEMANTIC_MODELS}
EmbeddingProgressCallback = Callable[[int, int, str], None]


def _registered_model_names() -> set[str]:
    from fastembed import TextEmbedding

    result: set[str] = set()
    for row in TextEmbedding.list_supported_models():
        if isinstance(row, dict):
            value = row.get("model")
        else:
            value = getattr(row, "model", None)
        if value:
            result.add(str(value).casefold())
    return result


def register_custom_model(spec: SemanticModelSpec) -> None:
    if spec.custom_pooling is None:
        return
    if spec.model_name.casefold() in _registered_model_names():
        return

    from fastembed import TextEmbedding
    from fastembed.common.model_description import ModelSource, PoolingType

    TextEmbedding.add_custom_model(
        model=spec.model_name,
        pooling=PoolingType(spec.custom_pooling),
        normalization=True,
        sources=ModelSource(hf=spec.model_name),
        dim=spec.dimensions,
        model_file=spec.model_file,
        description=(
            "BGE-M3 multilingue utilisé en représentation dense locale."
        ),
        license="mit",
        size_in_gb=spec.download_gib,
        additional_files=list(spec.additional_files),
    )


def _semantic_documents(
    items: list[dict],
    spec: SemanticModelSpec,
) -> list[dict]:
    return [
        {
            "kind": "arena_semantic",
            "id": str(item["id"]),
            "text": spec.text_prefix + hm._model_text(item),
        }
        for item in items
    ]


def prepare_model_embeddings(
    database: str | Path | None,
    items: list[dict],
    spec: SemanticModelSpec,
    *,
    cache_directory: str | Path | None = None,
    on_progress: EmbeddingProgressCallback | None = None,
) -> np.ndarray:
    register_custom_model(spec)
    matrix = cached_text_embeddings(
        database,
        _semantic_documents(items, spec),
        model_name=spec.model_name,
        cache_directory=cache_directory,
        batch_size=spec.batch_size,
        on_progress=on_progress,
    )
    if matrix.shape != (len(items), spec.dimensions):
        raise RuntimeError(
            f"{spec.label} a produit une matrice {matrix.shape}, "
            f"au lieu de ({len(items)}, {spec.dimensions})."
        )
    return matrix


def _json_list(value: object, *keys: str) -> list:
    current = value
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key)
        else:
            return []
    return current if isinstance(current, list) else []


def _named_entities(
    rows: list,
    *,
    id_key: str = "id",
    name_key: str = "name",
    limit: int | None = None,
) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    for row in rows[:limit]:
        if not isinstance(row, dict):
            continue
        name = str(row.get(name_key) or "").strip()
        if not name:
            continue
        identifier = str(row.get(id_key) or f"name:{name.casefold()}")
        result.append((identifier, name))
    return result


def _crew_entities(payload: dict, jobs: set[str]) -> list[tuple[str, str]]:
    rows = _json_list(payload, "credits", "crew")
    return _named_entities(
        [
            row
            for row in rows
            if str(row.get("job") or "").casefold()
            in {value.casefold() for value in jobs}
        ]
    )


def _candidate_item(tmdb_id: int, payload: dict) -> dict | None:
    media_type = str(payload.get("media_type") or "movie").casefold()
    if media_type == "tv" or (payload.get("name") and not payload.get("title")):
        return None
    title = str(payload.get("title") or payload.get("original_title") or "").strip()
    if not title:
        return None
    evidence = best_public_rating(
        tmdb_rating=payload.get("vote_average"),
        tmdb_votes=payload.get("vote_count"),
    )
    keywords = (
        _json_list(payload, "keywords", "keywords")
        or _json_list(payload, "keywords", "results")
    )
    countries = [
        str(row.get("name") or "").strip()
        for row in _json_list(payload, "production_countries")
        if isinstance(row, dict) and str(row.get("name") or "").strip()
    ]
    companies = [
        str(row.get("name") or "").strip()
        for row in _json_list(payload, "production_companies")
        if isinstance(row, dict) and str(row.get("name") or "").strip()
    ]
    return {
        "id": f"tmdb:{tmdb_id}",
        "tmdb_id": tmdb_id,
        "title": title,
        "title_type": "movie",
        "year": (
            int(str(payload.get("release_date"))[:4])
            if str(payload.get("release_date") or "")[:4].isdigit()
            else None
        ),
        "benchmark": evidence.adjusted_rating,
        "benchmark_raw": evidence.raw_rating,
        "benchmark_source": evidence.source,
        "benchmark_reliability": evidence.reliability,
        "votes": evidence.vote_count,
        "runtime": payload.get("runtime"),
        "release_date": payload.get("release_date"),
        "overview": payload.get("overview"),
        "tagline": payload.get("tagline"),
        "language": payload.get("original_language"),
        "countries": countries,
        "companies": companies,
        "metadata_status": "candidate_cache",
        "entities": {
            "genres": _named_entities(_json_list(payload, "genres")),
            "keywords": _named_entities(keywords),
            "directors": _crew_entities(payload, {"Director"}),
            "writers": _crew_entities(
                payload,
                {"Writer", "Screenplay", "Story"},
            ),
            "actors": _named_entities(
                _json_list(payload, "credits", "cast"),
                limit=5,
            ),
            "cinematographers": _crew_entities(
                payload,
                {"Director of Photography"},
            ),
            "composers": _crew_entities(
                payload,
                {"Original Music Composer"},
            ),
            "editors": _crew_entities(payload, {"Editor"}),
        },
        "catalogue_origin": "candidate_cache",
    }


def load_current_catalogue(
    database: str | Path | None,
    rated_items: list[dict],
) -> tuple[list[dict], dict]:
    initialize(database)
    rated_ids = {str(item["id"]) for item in rated_items}
    with connect(database) as connection:
        title_rows = connection.execute(
            """
            SELECT imdb_id, tmdb_id
            FROM titles
            WHERE tmdb_id IS NOT NULL
            """
        ).fetchall()
        cache_rows = connection.execute(
            """
            SELECT tmdb_id, payload_json, fetched_at
            FROM candidate_cache
            ORDER BY fetched_at DESC
            """
        ).fetchall()

    rated_tmdb = {
        int(row["tmdb_id"])
        for row in title_rows
        if row["imdb_id"] in rated_ids and row["tmdb_id"] is not None
    }
    catalogue = [
        {**item, "catalogue_origin": "rated_history"}
        for item in rated_items
    ]
    seen_tmdb = set(rated_tmdb)
    invalid = 0
    duplicate = 0
    for row in cache_rows:
        tmdb_id = int(row["tmdb_id"])
        if tmdb_id in seen_tmdb:
            duplicate += 1
            continue
        try:
            payload = json.loads(str(row["payload_json"]))
        except (TypeError, json.JSONDecodeError):
            invalid += 1
            continue
        imdb_id = str(
            (payload.get("external_ids") or {}).get("imdb_id") or ""
        )
        if imdb_id and imdb_id in rated_ids:
            seen_tmdb.add(tmdb_id)
            duplicate += 1
            continue
        candidate = _candidate_item(tmdb_id, payload)
        if candidate is None or is_series_type(candidate["title_type"]):
            invalid += 1
            continue
        seen_tmdb.add(tmdb_id)
        catalogue.append(candidate)

    return catalogue, {
        "rated_history": len(rated_items),
        "candidate_cache_movies": len(catalogue) - len(rated_items),
        "catalogue_total": len(catalogue),
        "duplicates_excluded": duplicate,
        "invalid_or_series_excluded": invalid,
        "unknown_candidates_are_not_negatives": True,
    }


def estimated_missing_download_gib(
    cache_directory: str | Path,
) -> float:
    root = Path(cache_directory)
    missing = 0.0
    for spec in SEMANTIC_MODELS:
        repositories = spec.cache_repositories or (spec.model_name,)
        markers = [
            root / f"models--{value.replace('/', '--')}"
            for value in repositories
        ]
        legacy = root / f"fast-{spec.model_name.rsplit('/', 1)[-1]}"
        if not any(marker.exists() for marker in [*markers, legacy]):
            missing += spec.download_gib
    return round(missing, 2)

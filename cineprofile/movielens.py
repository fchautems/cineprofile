from __future__ import annotations

import csv
import gc
import hashlib
import json
import re
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import httpx
import numpy as np
import pandas as pd
from scipy import sparse


MOVIELENS_VERSION = "ml-32m"
MOVIELENS_URL = (
    "https://files.grouplens.org/datasets/movielens/ml-32m.zip"
)
CACHE_VERSION = 1
RAW_CHECKSUMS = {
    "links.csv": "8f033867bcb4e6be8792b21468b4fa6e",
    "movies.csv": "0df90835c19151f9d819d0822e190797",
    "ratings.csv": "cf12b74f9ad4b94a011f079e26d4270a",
}
ProgressCallback = Callable[[int, int, str], None]


@dataclass(frozen=True)
class NeighborConfiguration:
    minimum_overlap: int
    shrinkage: float
    neighbors: int

    @property
    def label(self) -> str:
        return (
            f"overlap={self.minimum_overlap}, "
            f"shrinkage={self.shrinkage:g}, voisins={self.neighbors}"
        )


@dataclass
class MovieLensData:
    root: Path
    centered_ratings: sparse.csr_matrix
    user_means: np.ndarray
    movie_means: np.ndarray
    movie_counts: np.ndarray
    bayesian_scores: np.ndarray
    movie_years: np.ndarray
    movie_ids: np.ndarray
    titles: dict[int, str]
    imdb_to_movie: dict[str, int]
    global_mean: float
    bayesian_prior_count: float

    @property
    def user_count(self) -> int:
        return max(0, int(self.centered_ratings.shape[0]) - 1)

    @property
    def movie_count(self) -> int:
        return int(len(self.movie_ids))

    @property
    def rating_count(self) -> int:
        return int(self.centered_ratings.nnz)


@dataclass(frozen=True)
class NeighborPrediction:
    movie_ids: np.ndarray
    predictions: np.ndarray
    support_counts: np.ndarray
    support_weights: np.ndarray
    selected_neighbors: int
    positive_similarity_users: int
    mapped_profile_items: int


def canonical_imdb_id(value: object) -> str | None:
    raw = str(value or "").strip().lower()
    if raw.startswith("tt"):
        raw = raw[2:]
    digits = "".join(character for character in raw if character.isdigit())
    if not digits:
        return None
    return "tt" + digits.zfill(7)


def _md5(path: Path, *, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _raw_directory(root: str | Path) -> Path:
    return Path(root) / MOVIELENS_VERSION


def _raw_files_are_valid(raw: Path) -> bool:
    return all(
        (raw / name).is_file() and _md5(raw / name) == checksum
        for name, checksum in RAW_CHECKSUMS.items()
    )


def _safe_extract_dataset(archive: Path, destination: Path) -> None:
    wanted = {
        f"{MOVIELENS_VERSION}/links.csv": "links.csv",
        f"{MOVIELENS_VERSION}/movies.csv": "movies.csv",
        f"{MOVIELENS_VERSION}/ratings.csv": "ratings.csv",
    }
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as package:
        available = set(package.namelist())
        missing = sorted(set(wanted) - available)
        if missing:
            raise RuntimeError(
                "L’archive MovieLens est incomplète : " + ", ".join(missing)
            )
        for source, target_name in wanted.items():
            target = destination / target_name
            temporary = target.with_suffix(target.suffix + ".part")
            with package.open(source) as incoming, temporary.open("wb") as outgoing:
                shutil.copyfileobj(incoming, outgoing, length=4 * 1024 * 1024)
            temporary.replace(target)


def ensure_movielens_dataset(
    root: str | Path,
    *,
    on_progress: ProgressCallback | None = None,
) -> Path:
    root = Path(root)
    raw = _raw_directory(root)
    if _raw_files_are_valid(raw):
        if on_progress:
            on_progress(1, 1, "MovieLens 32M déjà présent et vérifié")
        return raw

    root.mkdir(parents=True, exist_ok=True)
    archive = root / f"{MOVIELENS_VERSION}.zip"
    partial = archive.with_suffix(".zip.part")
    downloaded = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={downloaded}-"} if downloaded else {}
    mode = "ab" if downloaded else "wb"
    try:
        with httpx.stream(
            "GET",
            MOVIELENS_URL,
            headers=headers,
            follow_redirects=True,
            timeout=httpx.Timeout(30.0, read=120.0),
        ) as response:
            if downloaded and response.status_code != 206:
                downloaded = 0
                mode = "wb"
            response.raise_for_status()
            remaining = int(response.headers.get("content-length") or 0)
            total = downloaded + remaining
            current = downloaded
            with partial.open(mode) as output:
                for chunk in response.iter_bytes(1024 * 1024):
                    output.write(chunk)
                    current += len(chunk)
                    if on_progress:
                        on_progress(
                            current,
                            total,
                            (
                                "Téléchargement MovieLens 32M · "
                                f"{current / (1024**2):.0f} Mio"
                            ),
                        )
        partial.replace(archive)
    except Exception as exc:
        raise RuntimeError(
            "Impossible de télécharger MovieLens 32M depuis GroupLens. "
            "Le téléchargement partiel est conservé pour une reprise : "
            f"{exc}"
        ) from exc

    if on_progress:
        on_progress(0, 1, "Extraction et vérification de MovieLens 32M")
    _safe_extract_dataset(archive, raw)
    invalid = [
        name
        for name, checksum in RAW_CHECKSUMS.items()
        if _md5(raw / name) != checksum
    ]
    if invalid:
        raise RuntimeError(
            "Les sommes de contrôle officielles échouent pour : "
            + ", ".join(invalid)
        )
    archive.unlink(missing_ok=True)
    if on_progress:
        on_progress(1, 1, "MovieLens 32M extrait et vérifié")
    return raw


def _line_count(path: Path) -> int:
    lines = 0
    last_byte = b""
    with path.open("rb") as stream:
        while block := stream.read(16 * 1024 * 1024):
            lines += block.count(b"\n")
            last_byte = block[-1:]
    if last_byte and last_byte != b"\n":
        lines += 1
    return max(0, lines - 1)


def _movie_metadata(
    raw: Path,
) -> tuple[np.ndarray, np.ndarray, dict[int, str], dict[str, int]]:
    titles: dict[int, str] = {}
    years_by_id: dict[int, int] = {}
    year_pattern = re.compile(r"\((\d{4})\)\s*$")
    with (raw / "movies.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as stream:
        for row in csv.DictReader(stream):
            movie_id = int(row["movieId"])
            title = str(row["title"])
            match = year_pattern.search(title)
            titles[movie_id] = title
            years_by_id[movie_id] = int(match.group(1)) if match else 0

    imdb_to_movie: dict[str, int] = {}
    with (raw / "links.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as stream:
        for row in csv.DictReader(stream):
            imdb_id = canonical_imdb_id(row.get("imdbId"))
            if imdb_id:
                imdb_to_movie[imdb_id] = int(row["movieId"])

    movie_ids = np.asarray(sorted(titles), dtype=np.int32)
    maximum = int(movie_ids[-1]) if len(movie_ids) else 0
    years = np.zeros(maximum + 1, dtype=np.int16)
    for movie_id, year in years_by_id.items():
        years[movie_id] = year
    return movie_ids, years, titles, imdb_to_movie


def _raw_signature(raw: Path) -> dict[str, dict[str, int | str]]:
    return {
        name: {
            "bytes": int((raw / name).stat().st_size),
            "mtime_ns": int((raw / name).stat().st_mtime_ns),
            "md5": checksum,
        }
        for name, checksum in RAW_CHECKSUMS.items()
    }


def _cache_is_current(cache: Path, raw: Path) -> bool:
    manifest = cache / "manifest.json"
    required = [
        manifest,
        cache / "centered_ratings.npz",
        cache / "statistics.npz",
        cache / "movies.json",
        cache / "links.json",
    ]
    if not all(path.is_file() for path in required):
        return False
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("cache_version") == CACHE_VERSION
        and payload.get("dataset") == MOVIELENS_VERSION
        and payload.get("raw_signature") == _raw_signature(raw)
    )


def build_movielens_cache(
    raw: str | Path,
    *,
    on_progress: ProgressCallback | None = None,
    chunksize: int = 1_000_000,
) -> Path:
    raw = Path(raw)
    cache = raw.parent / f"{MOVIELENS_VERSION}-cache-v{CACHE_VERSION}"
    if _cache_is_current(cache, raw):
        if on_progress:
            on_progress(1, 1, "Cache collaboratif MovieLens déjà prêt")
        return cache

    cache.mkdir(parents=True, exist_ok=True)
    temporary = cache / "building"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    ratings_path = raw / "ratings.csv"
    total = _line_count(ratings_path)
    if total <= 0:
        raise RuntimeError("ratings.csv ne contient aucune note.")

    rows_path = temporary / "rows.dat"
    columns_path = temporary / "columns.dat"
    values_path = temporary / "values.dat"
    rows = np.memmap(rows_path, dtype=np.int32, mode="w+", shape=(total,))
    columns = np.memmap(
        columns_path,
        dtype=np.int32,
        mode="w+",
        shape=(total,),
    )
    values = np.memmap(
        values_path,
        dtype=np.float32,
        mode="w+",
        shape=(total,),
    )

    offset = 0
    reader = pd.read_csv(
        ratings_path,
        usecols=["userId", "movieId", "rating"],
        dtype={"userId": "int32", "movieId": "int32", "rating": "float32"},
        chunksize=max(10_000, int(chunksize)),
    )
    for chunk in reader:
        size = len(chunk)
        end = offset + size
        rows[offset:end] = chunk["userId"].to_numpy(dtype=np.int32)
        columns[offset:end] = chunk["movieId"].to_numpy(dtype=np.int32)
        values[offset:end] = chunk["rating"].to_numpy(dtype=np.float32)
        offset = end
        if on_progress:
            on_progress(
                offset,
                total,
                f"Lecture des 32 millions de notes · {offset:,}/{total:,}",
            )
    if offset != total:
        raise RuntimeError(
            f"Nombre de notes incohérent : {offset:,} lues, {total:,} attendues."
        )

    maximum_user = int(np.max(rows))
    maximum_movie = int(np.max(columns))
    user_counts = np.bincount(rows, minlength=maximum_user + 1).astype(np.int32)
    user_sums = np.bincount(
        rows,
        weights=np.asarray(values, dtype=np.float64),
        minlength=maximum_user + 1,
    )
    user_means = np.zeros(maximum_user + 1, dtype=np.float32)
    valid_users = user_counts > 0
    user_means[valid_users] = (
        user_sums[valid_users] / user_counts[valid_users]
    ).astype(np.float32) * 2.0

    movie_counts = np.bincount(
        columns,
        minlength=maximum_movie + 1,
    ).astype(np.int32)
    movie_sums = np.bincount(
        columns,
        weights=np.asarray(values, dtype=np.float64),
        minlength=maximum_movie + 1,
    )
    movie_means = np.zeros(maximum_movie + 1, dtype=np.float32)
    valid_movies = movie_counts > 0
    movie_means[valid_movies] = (
        movie_sums[valid_movies] / movie_counts[valid_movies]
    ).astype(np.float32) * 2.0
    global_mean = float(np.mean(np.asarray(values, dtype=np.float64)) * 2.0)
    positive_counts = movie_counts[valid_movies]
    prior_count = float(np.quantile(positive_counts, 0.75))
    bayesian = np.full(maximum_movie + 1, global_mean, dtype=np.float32)
    weights = movie_counts[valid_movies] / (
        movie_counts[valid_movies] + prior_count
    )
    bayesian[valid_movies] = (
        global_mean
        + weights * (movie_means[valid_movies] - global_mean)
    ).astype(np.float32)

    step = 2_000_000
    for start in range(0, total, step):
        end = min(total, start + step)
        values[start:end] = (
            values[start:end] * 2.0 - user_means[rows[start:end]]
        )
        if on_progress:
            on_progress(
                end,
                total,
                f"Centrage des notes communautaires · {end:,}/{total:,}",
            )
    rows.flush()
    columns.flush()
    values.flush()

    matrix = sparse.csr_matrix(
        (
            np.asarray(values),
            (np.asarray(rows), np.asarray(columns)),
        ),
        shape=(maximum_user + 1, maximum_movie + 1),
        dtype=np.float32,
    )
    sparse.save_npz(
        cache / "centered_ratings.npz",
        matrix,
        compressed=True,
    )
    np.savez_compressed(
        cache / "statistics.npz",
        user_means=user_means,
        movie_means=movie_means,
        movie_counts=movie_counts,
        bayesian_scores=bayesian,
        global_mean=np.asarray([global_mean], dtype=np.float64),
        bayesian_prior_count=np.asarray([prior_count], dtype=np.float64),
    )
    movie_ids, years, titles, imdb_to_movie = _movie_metadata(raw)
    np.save(cache / "movie_ids.npy", movie_ids)
    np.save(cache / "movie_years.npy", years)
    (cache / "movies.json").write_text(
        json.dumps(
            {str(movie_id): title for movie_id, title in titles.items()},
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    (cache / "links.json").write_text(
        json.dumps(imdb_to_movie, separators=(",", ":")),
        encoding="utf-8",
    )
    manifest = {
        "cache_version": CACHE_VERSION,
        "dataset": MOVIELENS_VERSION,
        "raw_signature": _raw_signature(raw),
        "rating_count": total,
        "user_count": maximum_user,
        "movie_count": int(len(movie_ids)),
        "matrix_shape": list(matrix.shape),
        "global_mean_10_scale": global_mean,
        "bayesian_prior_count": prior_count,
    }
    (cache / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    del matrix, rows, columns, values
    gc.collect()
    shutil.rmtree(temporary, ignore_errors=True)
    if on_progress:
        on_progress(1, 1, "Cache collaboratif MovieLens construit")
    return cache


def load_movielens_data(
    root: str | Path,
    *,
    on_progress: ProgressCallback | None = None,
) -> MovieLensData:
    raw = ensure_movielens_dataset(root, on_progress=on_progress)
    cache = build_movielens_cache(raw, on_progress=on_progress)
    matrix = sparse.load_npz(cache / "centered_ratings.npz").tocsr()
    statistics = np.load(cache / "statistics.npz")
    titles = {
        int(movie_id): str(title)
        for movie_id, title in json.loads(
            (cache / "movies.json").read_text(encoding="utf-8")
        ).items()
    }
    imdb_to_movie = {
        str(imdb_id): int(movie_id)
        for imdb_id, movie_id in json.loads(
            (cache / "links.json").read_text(encoding="utf-8")
        ).items()
    }
    return MovieLensData(
        root=raw,
        centered_ratings=matrix,
        user_means=np.asarray(statistics["user_means"], dtype=np.float32),
        movie_means=np.asarray(statistics["movie_means"], dtype=np.float32),
        movie_counts=np.asarray(statistics["movie_counts"], dtype=np.int32),
        bayesian_scores=np.asarray(
            statistics["bayesian_scores"],
            dtype=np.float32,
        ),
        movie_years=np.load(cache / "movie_years.npy"),
        movie_ids=np.load(cache / "movie_ids.npy"),
        titles=titles,
        imdb_to_movie=imdb_to_movie,
        global_mean=float(statistics["global_mean"][0]),
        bayesian_prior_count=float(statistics["bayesian_prior_count"][0]),
    )


def mapped_profile(
    data: MovieLensData,
    items: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    movie_ids: list[int] = []
    ratings: list[float] = []
    seen: set[int] = set()
    for item in items:
        imdb_id = canonical_imdb_id(item.get("id") or item.get("imdb_id"))
        movie_id = data.imdb_to_movie.get(imdb_id or "")
        if (
            movie_id is None
            or movie_id in seen
            or movie_id >= data.centered_ratings.shape[1]
        ):
            continue
        seen.add(movie_id)
        movie_ids.append(movie_id)
        ratings.append(float(item["rating"]))
    return (
        np.asarray(movie_ids, dtype=np.int32),
        np.asarray(ratings, dtype=np.float32),
    )


def neighbor_predictions(
    data: MovieLensData,
    profile_items: list[dict],
    candidate_movie_ids: np.ndarray,
    configuration: NeighborConfiguration,
) -> NeighborPrediction:
    candidate_movie_ids = np.asarray(candidate_movie_ids, dtype=np.int32)
    movie_ids, ratings = mapped_profile(data, profile_items)
    profile_mean = float(
        np.mean([float(item["rating"]) for item in profile_items])
    )
    fallback = data.bayesian_scores[candidate_movie_ids].astype(
        np.float64,
        copy=True,
    )
    if not len(movie_ids):
        return NeighborPrediction(
            movie_ids=candidate_movie_ids,
            predictions=fallback,
            support_counts=np.zeros(len(candidate_movie_ids), dtype=np.int32),
            support_weights=np.zeros(len(candidate_movie_ids), dtype=np.float64),
            selected_neighbors=0,
            positive_similarity_users=0,
            mapped_profile_items=0,
        )

    residuals = ratings.astype(np.float64) - profile_mean
    profile_matrix = data.centered_ratings[:, movie_ids].tocsr()
    presence = profile_matrix.copy()
    presence.data = np.ones_like(presence.data, dtype=np.float32)
    overlaps = np.asarray(presence.sum(axis=1)).ravel()
    dots = np.asarray(profile_matrix @ residuals).ravel()
    squared_external = profile_matrix.copy()
    squared_external.data **= 2
    external_norm = np.asarray(squared_external.sum(axis=1)).ravel()
    target_norm = np.asarray(presence @ (residuals**2)).ravel()
    denominator = np.sqrt(external_norm * target_norm)
    similarities = np.zeros(len(overlaps), dtype=np.float64)
    valid = (
        (overlaps >= configuration.minimum_overlap)
        & (denominator > 1e-12)
    )
    similarities[valid] = dots[valid] / denominator[valid]
    similarities[valid] *= overlaps[valid] / (
        overlaps[valid] + float(configuration.shrinkage)
    )
    positive = np.flatnonzero(similarities > 0)
    positive_count = int(len(positive))
    if positive_count > configuration.neighbors:
        local = np.argpartition(
            similarities[positive],
            -configuration.neighbors,
        )[-configuration.neighbors :]
        neighbors = positive[local]
    else:
        neighbors = positive
    if len(neighbors):
        neighbors = neighbors[
            np.argsort(similarities[neighbors])[::-1]
        ]
    if not len(neighbors):
        return NeighborPrediction(
            movie_ids=candidate_movie_ids,
            predictions=fallback,
            support_counts=np.zeros(len(candidate_movie_ids), dtype=np.int32),
            support_weights=np.zeros(len(candidate_movie_ids), dtype=np.float64),
            selected_neighbors=0,
            positive_similarity_users=positive_count,
            mapped_profile_items=len(movie_ids),
        )

    weights = similarities[neighbors]
    candidate_matrix = data.centered_ratings[neighbors][
        :, candidate_movie_ids
    ].tocsr()
    numerator = np.asarray(candidate_matrix.T @ weights).ravel()
    candidate_presence = candidate_matrix.copy()
    candidate_presence.data = np.ones_like(
        candidate_presence.data,
        dtype=np.float32,
    )
    support_counts = np.asarray(candidate_presence.sum(axis=0)).ravel().astype(
        np.int32
    )
    support_weights = np.asarray(
        candidate_presence.T @ np.abs(weights)
    ).ravel()
    predictions = fallback
    supported = support_weights > 1e-12
    predictions[supported] = (
        profile_mean + numerator[supported] / support_weights[supported]
    )
    predictions = np.clip(predictions, 1.0, 10.0)
    return NeighborPrediction(
        movie_ids=candidate_movie_ids,
        predictions=predictions,
        support_counts=support_counts,
        support_weights=support_weights,
        selected_neighbors=int(len(neighbors)),
        positive_similarity_users=positive_count,
        mapped_profile_items=len(movie_ids),
    )

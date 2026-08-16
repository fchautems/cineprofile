from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from . import __version__


class RadarrError(RuntimeError):
    """Readable error raised when Radarr cannot fulfil a request."""


@dataclass(frozen=True)
class RadarrAddResult:
    movie_id: int | None
    already_present: bool
    search_command_id: int | None


class RadarrClient:
    def __init__(
        self,
        url: str,
        api_key: str,
        *,
        timeout: float = 20.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        base_url = url.strip().rstrip("/")
        if not base_url:
            raise ValueError("L’adresse de Radarr est vide.")
        if not api_key.strip():
            raise ValueError("La clé API de Radarr est vide.")
        self.client = httpx.Client(
            base_url=base_url,
            headers={
                "X-Api-Key": api_key.strip(),
                "Accept": "application/json",
                "User-Agent": f"CineProfile/{__version__}",
            },
            timeout=timeout,
            transport=transport,
        )

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "RadarrClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            response = self.client.request(method, path, **kwargs)
            response.raise_for_status()
        except httpx.RequestError as exc:
            raise RadarrError(f"Radarr est injoignable : {exc}") from exc
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:400].strip()
            suffix = f" — {detail}" if detail else ""
            raise RadarrError(
                f"Radarr a refusé la demande ({exc.response.status_code}){suffix}"
            ) from exc
        try:
            return response.json()
        except ValueError as exc:
            raise RadarrError("Radarr a renvoyé une réponse illisible.") from exc

    def status(self) -> dict:
        result = self._request("GET", "/api/v3/system/status")
        if not isinstance(result, dict):
            raise RadarrError("Le statut renvoyé par Radarr est invalide.")
        return result

    def root_folders(self) -> list[dict]:
        result = self._request("GET", "/api/v3/rootfolder")
        return result if isinstance(result, list) else []

    def quality_profiles(self) -> list[dict]:
        result = self._request("GET", "/api/v3/qualityprofile")
        return result if isinstance(result, list) else []

    def movies(self) -> list[dict]:
        result = self._request("GET", "/api/v3/movie")
        if not isinstance(result, list):
            raise RadarrError("La liste des films Radarr est invalide.")
        return [row for row in result if isinstance(row, dict)]

    def queue(self) -> list[dict]:
        result = self._request(
            "GET",
            "/api/v3/queue",
            params={"page": 1, "pageSize": 1000, "includeMovie": True},
        )
        if not isinstance(result, dict) or not isinstance(
            result.get("records"), list
        ):
            raise RadarrError("La file de téléchargement Radarr est invalide.")
        return [row for row in result["records"] if isinstance(row, dict)]

    @staticmethod
    def _queue_detail(queue_item: dict) -> str | None:
        if queue_item.get("errorMessage"):
            return str(queue_item["errorMessage"])
        messages: list[str] = []
        for group in queue_item.get("statusMessages") or []:
            if not isinstance(group, dict):
                continue
            messages.extend(str(row) for row in group.get("messages") or [])
        return " · ".join(messages[:3]) or None

    @classmethod
    def _derive_movie_state(
        cls,
        movie: dict | None,
        queue_item: dict | None,
    ) -> dict:
        if movie is None:
            return {
                "state": "missing",
                "detail": "Le film n’est plus présent dans Radarr.",
                "progress": None,
                "radarr_movie_id": None,
            }

        movie_id = int(movie["id"]) if movie.get("id") else None
        if movie.get("hasFile"):
            movie_file = movie.get("movieFile") or {}
            relative_path = str(movie_file.get("relativePath") or "").strip()
            size = int(movie.get("sizeOnDisk") or 0)
            details = [relative_path] if relative_path else []
            if size:
                details.append(f"{size / (1024 ** 3):.1f} Go")
            return {
                "state": "downloaded",
                "detail": " · ".join(details) or "Fichier importé par Radarr.",
                "progress": 100.0,
                "radarr_movie_id": movie_id,
            }

        if queue_item is not None:
            tracked_status = str(
                queue_item.get("trackedDownloadStatus") or ""
            ).casefold()
            tracked_state = str(
                queue_item.get("trackedDownloadState") or ""
            ).casefold()
            detail = cls._queue_detail(queue_item)
            if tracked_status == "error" or tracked_state in {
                "failed",
                "failedpending",
                "importblocked",
            }:
                return {
                    "state": "error",
                    "detail": detail or "Radarr signale un échec du téléchargement.",
                    "progress": None,
                    "radarr_movie_id": movie_id,
                }
            size = float(queue_item.get("size") or 0.0)
            size_left = float(queue_item.get("sizeleft") or 0.0)
            progress = (
                max(0.0, min(100.0, 100.0 * (size - size_left) / size))
                if size > 0
                else None
            )
            if tracked_state in {"importpending", "importing", "imported"}:
                detail = detail or "Téléchargement terminé, importation en cours."
            return {
                "state": "downloading",
                "detail": detail,
                "progress": progress,
                "radarr_movie_id": movie_id,
            }

        if movie.get("monitored") is False:
            return {
                "state": "unmonitored",
                "detail": "Le film est présent mais n’est plus monitoré.",
                "progress": None,
                "radarr_movie_id": movie_id,
            }
        if movie.get("lastSearchTime"):
            return {
                "state": "no_result",
                "detail": "Aucun téléchargement en cours ; Radarr reste monitoré.",
                "progress": None,
                "radarr_movie_id": movie_id,
            }
        if movie.get("isAvailable") is True or str(
            movie.get("status") or ""
        ).casefold() == "released":
            return {
                "state": "available",
                "detail": "Le film est sorti et surveillé par Radarr.",
                "progress": None,
                "radarr_movie_id": movie_id,
            }
        return {
            "state": "monitored",
            "detail": "Radarr surveille ce film.",
            "progress": None,
            "radarr_movie_id": movie_id,
        }

    def movie_states(self, tmdb_ids: set[int]) -> dict[int, dict]:
        """Return one honest technical state per requested TMDB movie."""
        wanted = {int(tmdb_id) for tmdb_id in tmdb_ids}
        movies = self.movies()
        queue = self.queue()
        movies_by_tmdb = {
            int(row["tmdbId"]): row
            for row in movies
            if row.get("tmdbId") and int(row["tmdbId"]) in wanted
        }
        queue_by_movie_id = {
            int(row["movieId"]): row
            for row in queue
            if row.get("movieId")
        }
        return {
            tmdb_id: self._derive_movie_state(
                movies_by_tmdb.get(tmdb_id),
                queue_by_movie_id.get(
                    int(movies_by_tmdb[tmdb_id]["id"])
                    if tmdb_id in movies_by_tmdb
                    and movies_by_tmdb[tmdb_id].get("id")
                    else -1
                ),
            )
            for tmdb_id in wanted
        }

    def all_movie_states(self) -> dict[int, dict]:
        """Return the current state of every movie carrying a TMDB id."""
        movies = self.movies()
        queue = self.queue()
        queue_by_movie_id = {
            int(row["movieId"]): row
            for row in queue
            if row.get("movieId")
        }
        states: dict[int, dict] = {}
        for movie in movies:
            if not movie.get("tmdbId"):
                continue
            movie_id = int(movie["id"]) if movie.get("id") else -1
            states[int(movie["tmdbId"])] = self._derive_movie_state(
                movie,
                queue_by_movie_id.get(movie_id),
            )
        return states

    def _existing_movie(self, tmdb_id: int) -> dict | None:
        result = self.movies()
        return next(
            (
                movie
                for movie in result
                if int(movie.get("tmdbId") or 0) == int(tmdb_id)
            ),
            None,
        )

    def _start_movie_search(self, movie_id: int) -> int | None:
        try:
            command = self._request(
                "POST",
                "/api/v3/command",
                json={"name": "MoviesSearch", "movieIds": [int(movie_id)]},
            )
        except RadarrError as exc:
            raise RadarrError(
                "Le film est bien dans Radarr, mais sa recherche n’a pas pu "
                f"être lancée : {exc}"
            ) from exc
        if not isinstance(command, dict):
            raise RadarrError(
                "Le film est bien dans Radarr, mais Radarr n’a pas confirmé "
                "le lancement de sa recherche."
            )
        return int(command["id"]) if command.get("id") else None

    def add_movie(
        self,
        tmdb_id: int,
        *,
        root_folder_path: str,
        quality_profile_id: int,
    ) -> RadarrAddResult:
        existing = self._existing_movie(tmdb_id)
        if existing is not None:
            if not existing.get("id"):
                raise RadarrError(
                    "Le film existe dans Radarr, mais son identifiant est absent."
                )
            movie_id = int(existing["id"])
            if existing.get("monitored") is False:
                monitored_movie = {**existing, "monitored": True}
                updated = self._request(
                    "PUT",
                    f"/api/v3/movie/{movie_id}",
                    params={"moveFiles": False},
                    json=monitored_movie,
                )
                if not isinstance(updated, dict):
                    raise RadarrError(
                        "Le film est dans Radarr, mais sa surveillance "
                        "n’a pas pu être réactivée."
                    )
            return RadarrAddResult(
                movie_id=movie_id,
                already_present=True,
                search_command_id=self._start_movie_search(movie_id),
            )

        lookup = self._request(
            "GET",
            "/api/v3/movie/lookup",
            params={"term": f"tmdb:{int(tmdb_id)}"},
        )
        if not isinstance(lookup, list):
            raise RadarrError("Radarr n’a pas renvoyé de résultat de recherche.")
        movie = next(
            (
                row
                for row in lookup
                if int(row.get("tmdbId") or 0) == int(tmdb_id)
            ),
            None,
        )
        if movie is None:
            raise RadarrError("Ce film est introuvable dans Radarr.")

        payload = dict(movie)
        payload.update(
            {
                "qualityProfileId": int(quality_profile_id),
                "rootFolderPath": str(root_folder_path),
                "monitored": True,
                "addOptions": {
                    # La recherche est lancée explicitement après l’ajout. Cela
                    # évite de dépendre du traitement implicite de cette option.
                    "searchForMovie": False,
                    "monitor": "movieOnly",
                },
            }
        )
        created = self._request("POST", "/api/v3/movie", json=payload)
        if not isinstance(created, dict):
            raise RadarrError("Radarr n’a pas confirmé l’ajout du film.")
        if not created.get("id"):
            raise RadarrError(
                "Le film a été ajouté, mais Radarr n’a pas renvoyé son "
                "identifiant."
            )
        movie_id = int(created["id"])
        return RadarrAddResult(
            movie_id=movie_id,
            already_present=False,
            search_command_id=self._start_movie_search(movie_id),
        )

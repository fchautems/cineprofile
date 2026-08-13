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

    def _existing_movie(self, tmdb_id: int) -> dict | None:
        result = self._request("GET", "/api/v3/movie")
        if not isinstance(result, list):
            return None
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
                "Le film a été ajouté, mais Radarr n’a pas renvoyé son identifiant."
            )
        movie_id = int(created["id"])
        return RadarrAddResult(
            movie_id=movie_id,
            already_present=False,
            search_command_id=self._start_movie_search(movie_id),
        )

from __future__ import annotations

from types import ModuleType
from typing import Any


EXPECTED_RECOMMENDATION_PROTOCOL = 17


class CineProfileVersionMismatch(RuntimeError):
    """Raised when Streamlit kept an older project module in memory."""


def loaded_recommendation_protocol(module: ModuleType) -> int:
    return int(getattr(module, "RECOMMENDATION_PROTOCOL", 1))


def ensure_recommendation_protocol(module: ModuleType) -> None:
    loaded = loaded_recommendation_protocol(module)
    if loaded != EXPECTED_RECOMMENDATION_PROTOCOL:
        raise CineProfileVersionMismatch(
            "La mise à jour a été copiée pendant que CineProfile tournait encore. "
            "Ferme complètement la fenêtre noire de CineProfile, puis relance "
            "« Lancer CineProfile.bat ». La base de films reste intacte."
        )


def unpack_recommendation_run(value: Any) -> tuple[list[dict], dict[str, Any]]:
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or not isinstance(value[0], list)
        or not isinstance(value[1], dict)
    ):
        raise CineProfileVersionMismatch(
            "L’interface et le moteur de recommandations ne sont pas de la même "
            "version. Ferme complètement la fenêtre noire de CineProfile, puis "
            "relance « Lancer CineProfile.bat »."
        )
    return value

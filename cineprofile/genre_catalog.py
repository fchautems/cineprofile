from __future__ import annotations


TMDB_EXCLUDABLE_GENRES = {
    "Action": 28,
    "Aventure": 12,
    "Animation": 16,
    "Comédie": 35,
    "Crime": 80,
    "Documentaire": 99,
    "Drame": 18,
    "Familial": 10751,
    "Fantastique": 14,
    "Histoire": 36,
    "Horreur": 27,
    "Musique": 10402,
    "Mystère": 9648,
    "Romance": 10749,
    "Science-Fiction": 878,
    "Téléfilm": 10770,
    "Thriller": 53,
    "Guerre": 10752,
    "Western": 37,
}

TMDB_TV_ONLY_GENRES = {
    10759,
    10762,
    10763,
    10764,
    10765,
    10766,
    10767,
    10768,
}

TMDB_MOVIE_ONLY_GENRES = {
    12,
    14,
    27,
    28,
    36,
    53,
    878,
    10402,
    10749,
    10752,
}


def genre_scope_label(value: object) -> str:
    genre_id = str(value)
    if not genre_id.isdigit():
        return "Films et séries"
    numeric_id = int(genre_id)
    if numeric_id in TMDB_TV_ONLY_GENRES:
        return "Séries TV"
    if numeric_id in TMDB_MOVIE_ONLY_GENRES:
        return "Films"
    return "Films et séries"

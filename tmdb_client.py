"""
tmdb_client.py (UPDATED)
------------------------
Fetches from TMDB:
  - Movie posters (official)
  - Movie trailers (YouTube embed links via /movie/{id}/videos)
  - General metadata (rating, backdrop, tagline)

TMDB API is 100% free for non-commercial use.
Docs: https://developer.themoviedb.org/docs
"""

from __future__ import annotations

import os
import requests
import streamlit as st
from functools import lru_cache
from typing import Optional


# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

TMDB_BASE_URL    = "https://api.themoviedb.org/3"
TMDB_POSTER_BASE = "https://image.tmdb.org/t/p/w500"
TMDB_BACK_BASE   = "https://image.tmdb.org/t/p/original"
TMDB_FALLBACK    = "https://placehold.co/300x450/141414/E50914?text=No+Poster"
REQUEST_TIMEOUT  = 5


def _get_api_key() -> Optional[str]:
    """Read TMDB API key from Streamlit secrets (cloud) or .env (local)."""
    try:
        return st.secrets["TMDB_API_KEY"]
    except (KeyError, FileNotFoundError):
        pass
    return os.getenv("TMDB_API_KEY")


def api_key_configured() -> bool:
    return _get_api_key() is not None


# ──────────────────────────────────────────────
# Core TMDB search
# ──────────────────────────────────────────────

@lru_cache(maxsize=256)
def search_tmdb(title: str, year: Optional[int] = None) -> Optional[dict]:
    """
    Search TMDB for a movie.
    Returns best-match dict or None if key is missing/request fails.
    """
    api_key = _get_api_key()
    if not api_key:
        return None

    params = {
        "api_key":        api_key,
        "query":          title,
        "language":       "en-US",
        "include_adult":  "false",
        "page":           1,
    }
    if year and year > 0:
        params["year"] = year

    try:
        resp = requests.get(
            f"{TMDB_BASE_URL}/search/movie",
            params=params,
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        return results[0] if results else None
    except (requests.RequestException, ValueError, IndexError):
        return None


# ──────────────────────────────────────────────
# Posters
# ──────────────────────────────────────────────

def get_poster_url(title: str, year: Optional[int] = None) -> str:
    """Return official TMDB poster URL. Falls back to styled placeholder."""
    movie = search_tmdb(title, year)
    if movie and movie.get("poster_path"):
        return f"{TMDB_POSTER_BASE}{movie['poster_path']}"
    return TMDB_FALLBACK


def get_backdrop_url(title: str, year: Optional[int] = None) -> str:
    """Return high-res backdrop image URL for hero sections."""
    movie = search_tmdb(title, year)
    if movie and movie.get("backdrop_path"):
        return f"{TMDB_BACK_BASE}{movie['backdrop_path']}"
    return ""


# ──────────────────────────────────────────────
# Trailers  [NEW]
# ──────────────────────────────────────────────

@lru_cache(maxsize=256)
def get_movie_trailer(title: str, year: Optional[int] = None) -> Optional[str]:
    """
    Fetch the official trailer YouTube embed URL for a movie.

    Calls:
      1. /search/movie → get TMDB ID
      2. /movie/{id}/videos → get YouTube trailer key

    Returns
    -------
    str  : YouTube embed URL  e.g. "https://www.youtube.com/embed/YoHD9XEInc0"
    None : if no trailer found or API key not set
    """
    api_key = _get_api_key()
    if not api_key:
        return None

    movie = search_tmdb(title, year)
    if not movie:
        return None

    tmdb_id = movie.get("id")
    if not tmdb_id:
        return None

    try:
        resp = requests.get(
            f"{TMDB_BASE_URL}/movie/{tmdb_id}/videos",
            params={"api_key": api_key, "language": "en-US"},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        videos = resp.json().get("results", [])

        # Prefer official YouTube trailers
        for video in videos:
            if (
                video.get("site") == "YouTube"
                and video.get("type") == "Trailer"
                and video.get("official", False)
            ):
                return f"https://www.youtube.com/embed/{video['key']}"

        # Fall back to any YouTube trailer
        for video in videos:
            if video.get("site") == "YouTube" and video.get("type") == "Trailer":
                return f"https://www.youtube.com/embed/{video['key']}"

        # Fall back to any YouTube video (teaser, clip)
        for video in videos:
            if video.get("site") == "YouTube":
                return f"https://www.youtube.com/embed/{video['key']}"

        return None

    except (requests.RequestException, ValueError):
        return None


# ──────────────────────────────────────────────
# Full metadata bundle
# ──────────────────────────────────────────────

def get_tmdb_metadata(title: str, year: Optional[int] = None) -> dict:
    """
    Return a metadata bundle with guaranteed keys (no KeyError risk).

    Keys returned:
        poster_url, backdrop_url, vote_average, vote_count,
        tmdb_id, trailer_url
    """
    defaults = {
        "poster_url":   TMDB_FALLBACK,
        "backdrop_url": "",
        "vote_average": 0.0,
        "vote_count":   0,
        "tmdb_id":      None,
        "trailer_url":  None,
    }

    movie = search_tmdb(title, year)
    if not movie:
        return defaults

    poster   = movie.get("poster_path")
    backdrop = movie.get("backdrop_path")
    tmdb_id  = movie.get("id")

    return {
        "poster_url":   f"{TMDB_POSTER_BASE}{poster}"   if poster   else TMDB_FALLBACK,
        "backdrop_url": f"{TMDB_BACK_BASE}{backdrop}"   if backdrop else "",
        "vote_average": movie.get("vote_average", 0.0),
        "vote_count":   movie.get("vote_count", 0),
        "tmdb_id":      tmdb_id,
        "trailer_url":  get_movie_trailer(title, year),
    }

"""
tmdb_client.py
--------------
Optional TMDB API client for fetching:
  - Movie poster images
  - Backdrop images
  - Official TMDB metadata (vote average, tagline, etc.)

Usage
-----
Set TMDB_API_KEY in your .env file or .streamlit/secrets.toml.
If the key is absent the app runs fine — posters simply won't appear.

TMDB API v3 docs: https://developer.themoviedb.org/reference/intro/getting-started
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

TMDB_BASE_URL   = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"   # 500 px wide posters
TMDB_FALLBACK   = "https://via.placeholder.com/300x450?text=No+Poster"

REQUEST_TIMEOUT = 5   # seconds – keeps the UI snappy


def _get_api_key() -> Optional[str]:
    """
    Read the TMDB API key from Streamlit secrets (cloud) or environment (local).
    Returns None if neither is configured.
    """
    # 1. Streamlit Cloud / secrets.toml
    try:
        return st.secrets["TMDB_API_KEY"]
    except (KeyError, FileNotFoundError):
        pass

    # 2. Local .env loaded via python-dotenv (or system env)
    return os.getenv("TMDB_API_KEY")


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

@lru_cache(maxsize=256)
def search_tmdb(title: str, year: Optional[int] = None) -> Optional[dict]:
    """
    Search TMDB for a movie by title and optional year.

    Returns the best-matching result dict from TMDB's /search/movie endpoint,
    or None if the API key is missing, the request fails, or no match is found.

    Result dict keys (most useful ones):
        id, title, release_date, overview, poster_path,
        backdrop_path, vote_average, vote_count, genre_ids
    """
    api_key = _get_api_key()
    if not api_key:
        return None

    params: dict = {
        "api_key": api_key,
        "query": title,
        "language": "en-US",
        "include_adult": "false",
        "page": 1,
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


def get_poster_url(title: str, year: Optional[int] = None) -> str:
    """
    Return a fully-qualified poster image URL for `title`.
    Falls back to a placeholder if TMDB is unavailable or has no image.
    """
    movie = search_tmdb(title, year)
    if movie and movie.get("poster_path"):
        return f"{TMDB_IMAGE_BASE}{movie['poster_path']}"
    return TMDB_FALLBACK


def get_tmdb_metadata(title: str, year: Optional[int] = None) -> dict:
    """
    Return enriched TMDB metadata for a movie.

    Returns a dict with guaranteed keys so callers never need to guard
    against missing keys:
        poster_url, backdrop_url, tagline,
        vote_average, vote_count, tmdb_id
    """
    defaults = {
        "poster_url":   TMDB_FALLBACK,
        "backdrop_url": "",
        "tagline":      "",
        "vote_average": 0.0,
        "vote_count":   0,
        "tmdb_id":      None,
    }

    movie = search_tmdb(title, year)
    if not movie:
        return defaults

    poster_path   = movie.get("poster_path")
    backdrop_path = movie.get("backdrop_path")

    return {
        "poster_url":   f"{TMDB_IMAGE_BASE}{poster_path}" if poster_path else TMDB_FALLBACK,
        "backdrop_url": f"{TMDB_IMAGE_BASE}{backdrop_path}" if backdrop_path else "",
        "tagline":      "",          # tagline needs a separate /movie/{id} call
        "vote_average": movie.get("vote_average", 0.0),
        "vote_count":   movie.get("vote_count", 0),
        "tmdb_id":      movie.get("id"),
    }


@lru_cache(maxsize=128)
def get_movie_details(tmdb_id: int) -> dict:
    """
    Fetch full movie details from TMDB /movie/{id}.
    Requires a TMDB API key.  Returns {} on failure.

    Useful extra fields: tagline, runtime, budget, revenue, homepage.
    """
    api_key = _get_api_key()
    if not api_key or not tmdb_id:
        return {}

    try:
        resp = requests.get(
            f"{TMDB_BASE_URL}/movie/{tmdb_id}",
            params={"api_key": api_key, "language": "en-US"},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()
    except (requests.RequestException, ValueError):
        return {}


def api_key_configured() -> bool:
    """Return True if a TMDB API key is available."""
    return _get_api_key() is not None

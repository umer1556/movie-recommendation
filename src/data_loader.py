"""
data_loader.py
--------------
Responsible for:
  - Loading movies.csv and ratings.csv from disk
  - Cleaning and validating the data
  - Building the combined text feature matrix for TF-IDF
  - Exposing cached loaders so Streamlit doesn't reload on every rerun
"""

import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

MOVIES_PATH  = os.path.join("data", "movies.csv")
RATINGS_PATH = os.path.join("data", "ratings.csv")

# Columns used to build the content feature string
FEATURE_COLUMNS = ["genre", "cast", "director", "keywords", "overview"]


# ──────────────────────────────────────────────
# Helper utilities
# ──────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """
    Lowercase, strip extra spaces, remove commas.
    Commas inside cast/genre strings break tokenisation if left in.
    """
    if not isinstance(text, str):
        return ""
    return text.lower().replace(",", " ").strip()


def _build_soup(row: pd.Series) -> str:
    """
    Combine all feature columns into one 'soup' string per movie.
    Genre and keywords are doubled so they carry more TF-IDF weight.
    """
    genre    = _clean_text(row.get("genre", ""))
    cast     = _clean_text(row.get("cast", ""))
    director = _clean_text(row.get("director", ""))
    keywords = _clean_text(row.get("keywords", ""))
    overview = _clean_text(row.get("overview", ""))

    # Repeat genre/keywords to boost their signal
    return f"{genre} {genre} {director} {cast} {keywords} {keywords} {overview}"


# ──────────────────────────────────────────────
# Public loaders  (cached so they run once)
# ──────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_movies(filepath: str = MOVIES_PATH) -> pd.DataFrame:
    """
    Load movies.csv, validate required columns, and add a 'soup' feature column.

    Returns a clean DataFrame indexed by position (0-based), with an additional
    lowercase-title column for fast lookup.

    Raises FileNotFoundError if the CSV is missing.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"movies.csv not found at '{filepath}'. "
            "Make sure the data/ directory is present."
        )

    df = pd.read_csv(filepath)

    required = {"movie_id", "title", "year", "genre", "cast",
                "director", "keywords", "overview"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"movies.csv is missing columns: {missing}")

    # Drop rows with no title
    df = df.dropna(subset=["title"]).reset_index(drop=True)

    # Fill NaN text fields with empty string so soup never breaks
    for col in FEATURE_COLUMNS:
        df[col] = df[col].fillna("")

    # Year as integer (coerce bad values to 0)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(int)

    # Build the combined feature string
    df["soup"] = df.apply(_build_soup, axis=1)

    # Lowercase title index for O(1) lookup later
    df["title_lower"] = df["title"].str.lower().str.strip()

    return df


@st.cache_data(show_spinner=False)
def load_ratings(filepath: str = RATINGS_PATH) -> pd.DataFrame:
    """
    Load ratings.csv.

    Expected columns: user_id, movie_id, rating (1-5 float).
    Returns an empty DataFrame (with correct columns) if the file doesn't exist,
    so the app degrades gracefully without collaborative filtering.
    """
    if not os.path.exists(filepath):
        return pd.DataFrame(columns=["user_id", "movie_id", "rating"])

    df = pd.read_csv(filepath)

    required = {"user_id", "movie_id", "rating"}
    if not required.issubset(df.columns):
        return pd.DataFrame(columns=["user_id", "movie_id", "rating"])

    # Clip ratings to a sensible range
    df["rating"] = df["rating"].clip(0.5, 5.0)

    return df


@st.cache_data(show_spinner=False)
def build_similarity_matrix(movies_df: pd.DataFrame):
    """
    Build a TF-IDF matrix over the 'soup' column and compute pairwise
    cosine similarity between all movies.

    Returns
    -------
    similarity_matrix : np.ndarray  shape (n_movies, n_movies)
        similarity_matrix[i][j] is the cosine similarity between movie i and j.
    """
    tfidf = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),   # unigrams + bigrams for better matching
        min_df=1,
        max_features=10_000,
    )

    tfidf_matrix = tfidf.fit_transform(movies_df["soup"])

    # Cosine similarity on sparse matrix – efficient for small datasets
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    return similarity_matrix


def build_user_movie_matrix(
    movies_df: pd.DataFrame,
    ratings_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Pivot ratings into a user × movie matrix (rows = users, cols = movie titles).
    Missing ratings are left as NaN (not zero – important for CF).

    Returns an empty DataFrame if ratings are unavailable.
    """
    if ratings_df.empty:
        return pd.DataFrame()

    # Map movie_id → title
    id_to_title = movies_df.set_index("movie_id")["title"].to_dict()
    ratings_df = ratings_df.copy()
    ratings_df["title"] = ratings_df["movie_id"].map(id_to_title)
    ratings_df = ratings_df.dropna(subset=["title"])

    matrix = ratings_df.pivot_table(
        index="user_id",
        columns="title",
        values="rating",
        aggfunc="mean",  # average if a user rated the same movie twice
    )
    return matrix

"""
recommender.py
--------------
Two recommendation strategies:

1. Content-Based Filtering (CBF)
   - Uses the pre-computed cosine similarity matrix from data_loader.
   - get_recommendations_by_movie()   – single seed title
   - get_recommendations_by_titles()  – multiple seed titles (union/average)

2. User-Based Collaborative Filtering (CF)  [optional / degrades gracefully]
   - Pearson correlation between users on the ratings matrix.
   - get_recommendations_for_user()
   - Falls back to CBF if ratings data is too sparse.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


# ──────────────────────────────────────────────
# Content-Based Filtering
# ──────────────────────────────────────────────

def get_recommendations_by_movie(
    title: str,
    movies_df: pd.DataFrame,
    similarity_matrix: np.ndarray,
    n: int = 10,
) -> pd.DataFrame:
    """
    Return the top-n most similar movies to `title` using cosine similarity.

    Parameters
    ----------
    title            : Movie title (case-insensitive, partial match supported).
    movies_df        : DataFrame produced by data_loader.load_movies().
    similarity_matrix: np.ndarray from data_loader.build_similarity_matrix().
    n                : Number of recommendations to return.

    Returns
    -------
    DataFrame with columns: title, year, genre, overview, similarity_score
    Sorted descending by similarity_score.

    Raises
    ------
    ValueError if the title is not found even with fuzzy matching.
    """
    idx = _find_movie_index(title, movies_df)
    if idx is None:
        raise ValueError(
            f"Movie '{title}' not found. "
            "Try a different spelling or pick from the dropdown."
        )

    # Get similarity scores for every movie vs. the seed
    sim_scores = list(enumerate(similarity_matrix[idx]))

    # Sort descending by score, skip the first result (the seed itself)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [s for s in sim_scores if s[0] != idx][:n]

    movie_indices = [s[0] for s in sim_scores]
    scores        = [round(s[1], 4) for s in sim_scores]

    result = movies_df.iloc[movie_indices][
        ["title", "year", "genre", "overview", "movie_id"]
    ].copy()
    result["similarity_score"] = scores

    return result.reset_index(drop=True)


def get_recommendations_by_titles(
    titles: list[str],
    movies_df: pd.DataFrame,
    similarity_matrix: np.ndarray,
    n: int = 10,
) -> pd.DataFrame:
    """
    Generate recommendations from multiple seed movies.

    Strategy: average the similarity vectors of all valid seeds,
    then rank by that averaged score. This gives a 'blended taste' result.

    Movies that are in the seed list are excluded from results.
    """
    valid_indices: list[int] = []
    for t in titles:
        idx = _find_movie_index(t, movies_df)
        if idx is not None:
            valid_indices.append(idx)

    if not valid_indices:
        raise ValueError("None of the provided titles were found in the dataset.")

    # Average similarity row across all seed movies
    avg_scores = np.mean(similarity_matrix[valid_indices, :], axis=0)

    # Build ranked list, excluding seeds
    sim_scores = [
        (i, float(avg_scores[i]))
        for i in range(len(avg_scores))
        if i not in valid_indices
    ]
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[:n]

    movie_indices = [s[0] for s in sim_scores]
    scores        = [round(s[1], 4) for s in sim_scores]

    result = movies_df.iloc[movie_indices][
        ["title", "year", "genre", "overview", "movie_id"]
    ].copy()
    result["similarity_score"] = scores

    return result.reset_index(drop=True)


# ──────────────────────────────────────────────
# Collaborative Filtering (User-Based)
# ──────────────────────────────────────────────

def get_recommendations_for_user(
    user_id: str,
    user_movie_matrix: pd.DataFrame,
    movies_df: pd.DataFrame,
    similarity_matrix: np.ndarray,
    n: int = 10,
    min_common_movies: int = 2,
) -> pd.DataFrame:
    """
    User-based CF using Pearson correlation.

    Algorithm:
      1. Find users most correlated with `user_id`.
      2. Collect movies they rated highly that `user_id` hasn't seen.
      3. Score each candidate movie as a weighted average of similar
         users' ratings (weight = Pearson correlation).

    Falls back to CBF (using the user's highest-rated movies as seeds)
    if the matrix is too sparse or the user has no close neighbours.

    Parameters
    ----------
    user_id           : Must exist in user_movie_matrix.index.
    user_movie_matrix : pivot from data_loader.build_user_movie_matrix().
    movies_df         : Full movie DataFrame.
    similarity_matrix : Pre-computed cosine similarity (for CBF fallback).
    n                 : Number of recommendations.
    min_common_movies : Minimum co-rated movies to consider two users similar.

    Returns
    -------
    DataFrame with columns: title, year, genre, overview, predicted_rating
    """
    if user_movie_matrix.empty or user_id not in user_movie_matrix.index:
        # Graceful CBF fallback: return popular/high-variety recommendations
        return _cbf_fallback(movies_df, similarity_matrix, n)

    user_ratings = user_movie_matrix.loc[user_id]
    seen_movies  = set(user_ratings.dropna().index)

    # ── Compute Pearson correlation with every other user ──
    correlations: dict[str, float] = {}
    for other_user in user_movie_matrix.index:
        if other_user == user_id:
            continue

        other_ratings = user_movie_matrix.loc[other_user]

        # Only compare on movies both users have rated
        common = user_ratings.index[
            user_ratings.notna() & other_ratings.notna()
        ]
        if len(common) < min_common_movies:
            continue

        corr = user_ratings[common].corr(other_ratings[common])
        if pd.notna(corr) and corr > 0:   # only positive neighbours
            correlations[other_user] = corr

    if not correlations:
        return _cbf_fallback(movies_df, similarity_matrix, n, seen_movies)

    # ── Weighted score for unseen movies ──
    scores: dict[str, list] = {}   # movie → [weighted_rating, weight_sum]
    for neighbour, weight in correlations.items():
        neighbour_ratings = user_movie_matrix.loc[neighbour]
        for movie, rating in neighbour_ratings.items():
            if pd.isna(rating) or movie in seen_movies:
                continue
            if movie not in scores:
                scores[movie] = [0.0, 0.0]
            scores[movie][0] += weight * rating
            scores[movie][1] += weight

    if not scores:
        return _cbf_fallback(movies_df, similarity_matrix, n, seen_movies)

    predicted = {
        movie: vals[0] / vals[1]
        for movie, vals in scores.items()
        if vals[1] > 0
    }

    top_movies = sorted(predicted, key=predicted.get, reverse=True)[:n]

    result = movies_df[movies_df["title"].isin(top_movies)][
        ["title", "year", "genre", "overview", "movie_id"]
    ].copy()
    result["predicted_rating"] = result["title"].map(predicted).round(2)
    result = result.sort_values("predicted_rating", ascending=False)

    return result.reset_index(drop=True)


# ──────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────

def _find_movie_index(title: str, movies_df: pd.DataFrame) -> Optional[int]:
    """
    Find DataFrame row index for a movie title.
    Tries exact match first, then case-insensitive, then substring.
    Returns None if nothing matches.
    """
    title_lower = title.lower().strip()

    # Exact (case-insensitive) match
    exact = movies_df[movies_df["title_lower"] == title_lower]
    if not exact.empty:
        return exact.index[0]

    # Substring match (handles 'Dark Knight' → 'The Dark Knight')
    partial = movies_df[movies_df["title_lower"].str.contains(title_lower, regex=False)]
    if not partial.empty:
        return partial.index[0]

    return None


def _cbf_fallback(
    movies_df: pd.DataFrame,
    similarity_matrix: np.ndarray,
    n: int,
    exclude_titles: set | None = None,
) -> pd.DataFrame:
    """
    When CF can't run, return diverse high-similarity recommendations
    by using the first movie as a soft seed (works for demo purposes).
    In a real system you'd use popularity rank here.
    """
    exclude_titles = exclude_titles or set()
    candidates = movies_df[~movies_df["title"].isin(exclude_titles)]
    return candidates[["title", "year", "genre", "overview", "movie_id"]].head(n).copy()


def search_movies(query: str, movies_df: pd.DataFrame, max_results: int = 8) -> pd.DataFrame:
    """
    Simple title/genre/keyword search for the search-bar autocomplete.
    Returns rows whose title or genre contains `query` (case-insensitive).
    """
    q = query.lower().strip()
    if not q:
        return pd.DataFrame()

    mask = (
        movies_df["title_lower"].str.contains(q, regex=False) |
        movies_df["genre"].str.lower().str.contains(q, regex=False) |
        movies_df["keywords"].str.lower().str.contains(q, regex=False)
    )
    return movies_df[mask][["title", "year", "genre"]].head(max_results)

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

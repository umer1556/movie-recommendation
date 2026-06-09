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
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df.dropna(subset=["rating"], inplace=True)

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

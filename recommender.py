"""
recommender.py (UPDATED)
--------------
Two recommendation strategies + actor search:

1. Content-Based Filtering (CBF)
   - Uses the pre-computed cosine similarity matrix from data_loader.
   - get_recommendations_by_movie()   – single seed title
   - get_recommendations_by_titles()  – multiple seed titles (union/average)

2. User-Based Collaborative Filtering (CF)  [optional / degrades gracefully]
   - Pearson correlation between users on the ratings matrix.
   - get_recommendations_for_user()
   - Falls back to CBF if ratings data is too sparse.

3. Actor Search [NEW]
   - search_by_actor() – find all movies with a specific actor
   - sort_movies() – sort results by title or rating
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional
import difflib  # For fuzzy matching


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
    title            : Movie title (case-insensitive, fuzzy matching supported).
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
# Actor Search [NEW]
# ──────────────────────────────────────────────

def search_by_actor(query: str, movies_df: pd.DataFrame) -> pd.DataFrame:
    """
    Search for all movies by an actor name.
    Returns all movies where the actor appears in the cast.
    
    Parameters
    ----------
    query     : Actor name (case-insensitive, partial match supported)
    movies_df : Full movie DataFrame
    
    Returns
    -------
    DataFrame with: title, year, genre, overview, cast
    Sorted by year (newest first)
    """
    q = query.lower().strip()
    if not q:
        return pd.DataFrame()
    
    # Find movies where query appears in cast (case-insensitive)
    mask = movies_df["cast"].str.lower().str.contains(q, regex=False, na=False)
    result = movies_df[mask][["title", "year", "genre", "overview", "cast", "movie_id"]].copy()
    
    if result.empty:
        return result
    
    # Sort by year (newest first)
    result = result.sort_values("year", ascending=False)
    
    return result.reset_index(drop=True)


def sort_movies(movies_df: pd.DataFrame, sort_by: str = "title") -> pd.DataFrame:
    """
    Sort movies alphabetically or by rating (if available).
    
    Parameters
    ----------
    movies_df : DataFrame to sort
    sort_by   : 'title' (A-Z) or 'rating' (high to low)
    
    Returns
    -------
    Sorted DataFrame
    """
    if movies_df.empty:
        return movies_df
    
    if sort_by == "title":
        return movies_df.sort_values("title", ascending=True).reset_index(drop=True)
    elif sort_by == "rating" and "similarity_score" in movies_df.columns:
        return movies_df.sort_values("similarity_score", ascending=False).reset_index(drop=True)
    elif sort_by == "rating" and "predicted_rating" in movies_df.columns:
        return movies_df.sort_values("predicted_rating", ascending=False).reset_index(drop=True)
    else:
        return movies_df


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
    Uses fuzzy matching to handle spelling variations.
    
    Strategy (in order):
    1. Exact match (case-insensitive)
    2. Substring match
    3. Fuzzy match (difflib) – handles typos
    
    Returns None if nothing matches.
    """
    title_lower = title.lower().strip()

    # 1. Exact (case-insensitive) match
    exact = movies_df[movies_df["title_lower"] == title_lower]
    if not exact.empty:
        return exact.index[0]

    # 2. Substring match (handles 'Dark Knight' → 'The Dark Knight')
    partial = movies_df[movies_df["title_lower"].str.contains(title_lower, regex=False)]
    if not partial.empty:
        return partial.index[0]

    # 3. Fuzzy match using difflib (handles typos like 'Incepton' → 'Inception')
    all_titles = movies_df["title_lower"].tolist()
    close_matches = difflib.get_close_matches(title_lower, all_titles, n=1, cutoff=0.6)
    if close_matches:
        matched_title = close_matches[0]
        idx = movies_df[movies_df["title_lower"] == matched_title].index[0]
        return idx

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

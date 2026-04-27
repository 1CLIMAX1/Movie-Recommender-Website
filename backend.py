from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from difflib import get_close_matches
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import requests

# Load API key securely from environment variable
API_KEY = os.getenv("OMDB_API_KEY")

app = Flask(__name__)
CORS(app)

# Safe file paths (important for deployment)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

df = pickle.load(open(os.path.join(BASE_DIR, "movies.pkl"), "rb"))
tfidf_matrix = pickle.load(open(os.path.join(BASE_DIR, "tfidf.pkl"), "rb"))

numeric_matrix = df[["averageRating", "numVotes", "runtimeMinutes"]].fillna(0).to_numpy()
numeric_matrix = StandardScaler().fit_transform(numeric_matrix)

@lru_cache(maxsize=256)
def get_movie_details(title):
    if not API_KEY:
        return {
            "poster": "https://via.placeholder.com/150",
            "rating": "N/A",
            "director": "N/A",
            "actors": "N/A",
            "released": "N/A",
            "genre": "N/A",
            "boxoffice": "N/A"
        }

    try:
        url = f"http://www.omdbapi.com/?t={title}&apikey={API_KEY}"
        response = requests.get(url, timeout=3)
        data = response.json()

        if data.get("Response", "False") != "True":
            raise ValueError("OMDB lookup failed")

        poster = data.get("Poster", "")
        if poster == "N/A":
            poster = "https://via.placeholder.com/150"

        rating = data.get("imdbRating", "N/A")
        director = data.get("Director", "N/A")
        actors = data.get("Actors", "N/A")
        released = data.get("Released", "N/A")
        genre = data.get("Genre", "N/A")
        boxoffice = data.get("BoxOffice", "N/A")

        if actors != "N/A":
            actors = ", ".join(actors.split(",")[:3])

        return {
            "poster": poster,
            "rating": rating,
            "director": director,
            "actors": actors,
            "released": released,
            "genre": genre,
            "boxoffice": boxoffice
        }

    except Exception:
        return {
            "poster": "https://via.placeholder.com/150",
            "rating": "N/A",
            "director": "N/A",
            "actors": "N/A",
            "released": "N/A",
            "genre": "N/A",
            "boxoffice": "N/A"
        }

def _find_movie_index(title):
    title = title.strip().lower()

    exact_matches = df[df['primaryTitle'].str.lower().str.strip() == title]
    if not exact_matches.empty:
        return exact_matches.sort_values(
            by=['numVotes', 'averageRating'], ascending=False
        ).index[0]

    contains = df[df['primaryTitle'].str.lower().str.contains(re.escape(title), na=False)]
    if not contains.empty:
        starts = contains[contains['primaryTitle'].str.lower().str.startswith(title)]
        if not starts.empty:
            return starts.sort_values(
                by=['numVotes', 'averageRating'], ascending=False
            ).index[0]
        return contains.sort_values(
            by=['numVotes', 'averageRating'], ascending=False
        ).index[0]

    close = get_close_matches(title, df['primaryTitle'].str.lower().tolist(), n=1, cutoff=0.55)
    if close:
        match = df[df['primaryTitle'].str.lower() == close[0]]
        if not match.empty:
            return match.index[0]

    return None


def recommend(title):
    idx = _find_movie_index(title)
    if idx is None:
        return []

    # 🔹 Text similarity (PRIMARY signal)
    tfidf_scores = cosine_similarity(
        tfidf_matrix[idx:idx+1], tfidf_matrix
    ).flatten()

    # 🔹 Get input movie genres when available
    genres_column = 'genres' if 'genres' in df.columns else None

    if genres_column:
        input_genres = str(df[genres_column].iloc[idx])
        candidate_indices = [
            i for i in range(len(df))
            if i != idx and any(
                g.strip() in str(df[genres_column].iloc[i])
                for g in input_genres.split(',')
                if g.strip()
            )
        ]
    else:
        candidate_indices = [i for i in range(len(df)) if i != idx]

    # 🔹 Rank candidates using similarity + quality boost
    ranked = sorted(
        candidate_indices,
        key=lambda i: (
            0.85 * tfidf_scores[i] +                      # similarity
            0.10 * df['averageRating'].iloc[i] +          # rating boost (scaled values)
            0.05 * max(df['numVotes'].iloc[i], 0)         # popularity boost (scaled values)
        ),
        reverse=True
    )

    # 🔹 Final filtering (remove garbage)
    top_indices = [
        i for i in ranked
        if df['numVotes'].iloc[i] > 0 and df['averageRating'].iloc[i] > 0
    ]

    if not top_indices:
        top_indices = ranked[:5]
    else:
        top_indices = top_indices[:5]

    results = []

    with ThreadPoolExecutor(max_workers=min(5, len(top_indices))) as executor:
        futures = {
            executor.submit(get_movie_details, df['primaryTitle'].iloc[i]): i
            for i in top_indices
        }
        for future in as_completed(futures):
            i = futures[future]
            movie_title = df['primaryTitle'].iloc[i]
            details = future.result()
            results.append({
                "title": movie_title,
                "poster": details["poster"],
                "rating": details["rating"],
                "director": details["director"],
                "actors": details["actors"],
                "released": details["released"],
                "genre": details["genre"],
                "boxoffice": details["boxoffice"]
            })

    return results

@app.route("/recommend", methods=["POST", "OPTIONS"])
def recommend_api():
    if request.method == "OPTIONS":
        return jsonify({}), 200

    data = request.get_json()

    if not data or "movie" not in data:
        return jsonify({"error": "No movie provided"}), 400

    movie = data["movie"]
    try:
        results = recommend(movie)
    except Exception as e:
        return jsonify({"error": "Server error", "details": str(e)}), 500

    return jsonify(results)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
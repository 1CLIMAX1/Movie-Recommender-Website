from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import os
import re
from difflib import get_close_matches
from sklearn.metrics.pairwise import cosine_similarity
import requests

# Load API key securely from environment variable
API_KEY = os.getenv("OMDB_API_KEY")

app = Flask(__name__)
CORS(app)

# Safe file paths (important for deployment)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

df = pickle.load(open(os.path.join(BASE_DIR, "movies.pkl"), "rb"))
tfidf_matrix = pickle.load(open(os.path.join(BASE_DIR, "tfidf.pkl"), "rb"))

def get_movie_details(title):
    try:
        url = f"http://www.omdbapi.com/?t={title}&apikey={API_KEY}"
        response = requests.get(url, timeout=5)
        data = response.json()

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

    sim_scores = cosine_similarity(tfidf_matrix[idx:idx+1], tfidf_matrix).flatten()
    sim_indices = [i for i in sim_scores.argsort()[::-1] if i != idx][:5]

    results = []

    for i in sim_indices:
        movie_title = df['primaryTitle'].iloc[i]
        details = get_movie_details(movie_title)

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

@app.route("/recommend", methods=["POST"])
def recommend_api():
    data = request.get_json()

    if not data or "movie" not in data:
        return jsonify({"error": "No movie provided"}), 400

    movie = data["movie"]
    results = recommend(movie)

    return jsonify(results)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
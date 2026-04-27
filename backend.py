from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from difflib import get_close_matches
import numpy as np
from sklearn.preprocessing import StandardScaler
import requests

API_KEY = os.getenv("OMDB_API_KEY")

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

df = pickle.load(open(os.path.join(BASE_DIR, "movies.pkl"), "rb"))
NEIGHBOURS = pickle.load(open(os.path.join(BASE_DIR, "neighbours.pkl"), "rb"))
print(f"Neighbour lookup loaded ({len(NEIGHBOURS)} entries).", flush=True)

TITLES_LOWER = df['primaryTitle'].str.lower().str.strip().tolist()
_rec_cache: dict = {}


@lru_cache(maxsize=512)
def get_movie_details(title: str) -> dict:
    _default = {
        "poster": "https://placehold.co/200x300/1a1a1a/666?text=No+Poster",
        "rating": "N/A", "director": "N/A", "actors": "N/A",
        "released": "N/A", "genre": "N/A", "boxoffice": "N/A"
    }
    if not API_KEY:
        return _default
    try:
        url = f"http://www.omdbapi.com/?t={requests.utils.quote(title)}&apikey={API_KEY}"
        resp = requests.get(url, timeout=4)
        data = resp.json()
        if data.get("Response") != "True":
            return _default

        poster = data.get("Poster") or ""
        if poster in ("", "N/A"):
            poster = _default["poster"]

        actors = data.get("Actors", "N/A")
        if actors != "N/A":
            actors = ", ".join(actors.split(",")[:3])

        return {
            "poster":    poster,
            "rating":    data.get("imdbRating", "N/A"),
            "director":  data.get("Director",   "N/A"),
            "actors":    actors,
            "released":  data.get("Released",   "N/A"),
            "genre":     data.get("Genre",      "N/A"),
            "boxoffice": data.get("BoxOffice",  "N/A"),
        }
    except Exception:
        return _default


def _find_movie_index(title: str):
    title = title.strip().lower()

    exact_rows = df[[t == title for t in TITLES_LOWER]]
    if not exact_rows.empty:
        return exact_rows.sort_values(
            by=["numVotes", "averageRating"], ascending=False
        ).index[0]

    contains_rows = df[[title in t for t in TITLES_LOWER]]
    if not contains_rows.empty:
        starts = contains_rows[
            [TITLES_LOWER[i].startswith(title) for i in contains_rows.index]
        ]
        pool = starts if not starts.empty else contains_rows
        return pool.sort_values(
            by=["numVotes", "averageRating"], ascending=False
        ).index[0]

    close = get_close_matches(title, TITLES_LOWER, n=1, cutoff=0.55)
    if close:
        match = df[[t == close[0] for t in TITLES_LOWER]]
        if not match.empty:
            return match.index[0]

    return None


def recommend(title: str) -> list:
    key = title.strip().lower()
    print(f"[1] recommend() called: '{key}'", flush=True)

    if key in _rec_cache:
        print(f"[2] Cache hit — returning instantly", flush=True)
        return _rec_cache[key]

    t1 = time.time()
    idx = _find_movie_index(title)
    print(f"[2] Title matching: {time.time()-t1:.2f}s  idx={idx}", flush=True)

    if idx is None:
        return []

    t2 = time.time()
    neighbour_indices = NEIGHBOURS.get(idx, [])
    print(f"[3] Neighbour lookup: {time.time()-t2:.2f}s  found={len(neighbour_indices)}", flush=True)

    votes   = df['numVotes'].to_numpy()
    ratings = df['averageRating'].to_numpy()
    top_indices = [i for i in neighbour_indices if votes[i] > 0 and ratings[i] > 0][:5]
    if not top_indices:
        top_indices = neighbour_indices[:5]
    print(f"[4] Top indices: {top_indices}", flush=True)

    t3 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(get_movie_details, df['primaryTitle'].iloc[i]): i
            for i in top_indices
        }
        for future in as_completed(futures):
            i = futures[future]
            details = future.result()
            print(f"[5] OMDB done for: {df['primaryTitle'].iloc[i]}", flush=True)
            results.append({
                "title":     df['primaryTitle'].iloc[i],
                "poster":    details["poster"],
                "rating":    details["rating"],
                "director":  details["director"],
                "actors":    details["actors"],
                "released":  details["released"],
                "genre":     details["genre"],
                "boxoffice": details["boxoffice"],
            })
    print(f"[6] OMDB total: {time.time()-t3:.2f}s", flush=True)

    _rec_cache[key] = results
    return results


@app.route("/recommend", methods=["POST", "OPTIONS"])
def recommend_api():
    print("REQUEST RECEIVED", flush=True)
    if request.method == "OPTIONS":
        return jsonify({}), 200

    data = request.get_json()
    print(f"JSON parsed: {data}", flush=True)

    if not data or "movie" not in data:
        return jsonify({"error": "No movie provided"}), 400

    try:
        results = recommend(data["movie"])
    except Exception as e:
        return jsonify({"error": "Server error", "details": str(e)}), 500

    return jsonify(results)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
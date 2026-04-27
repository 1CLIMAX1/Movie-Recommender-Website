from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from difflib import get_close_matches
import requests

API_KEY = os.getenv("OMDB_API_KEY")

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── Load data once at startup ────────────────────────────────────────────────
df         = pickle.load(open(os.path.join(BASE_DIR, "movies.pkl"), "rb"))
NEIGHBOURS = pickle.load(open(os.path.join(BASE_DIR, "neighbours.pkl"), "rb"))

# ─── Pre-build lower-cased title list for fast matching ──────────────────────
TITLES_LOWER = df['primaryTitle'].str.lower().str.strip().tolist()

# ─── Pre-extract numpy arrays to avoid repeated .to_numpy() calls ─────────────
VOTES   = df['numVotes'].to_numpy()
RATINGS = df['averageRating'].to_numpy()

# ─── Result-level cache ───────────────────────────────────────────────────────
_rec_cache: dict = {}

print(f"Ready — {len(NEIGHBOURS)} movies indexed.", flush=True)


# ─── OMDB ─────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=512)
def get_movie_details(title: str) -> dict:
    _default = {
        "poster":    "https://placehold.co/200x300/1a1a1a/666?text=No+Poster",
        "rating":    "N/A",
        "director":  "N/A",
        "actors":    "N/A",
        "released":  "N/A",
        "genre":     "N/A",
        "boxoffice": "N/A",
    }
    if not API_KEY:
        return _default
    try:
        url  = f"http://www.omdbapi.com/?t={requests.utils.quote(title)}&apikey={API_KEY}"
        data = requests.get(url, timeout=5).json()
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


# ─── Title matching ───────────────────────────────────────────────────────────
def _find_movie_index(title: str):
    t = title.strip().lower()

    # 1. Exact
    exact = df[[s == t for s in TITLES_LOWER]]
    if not exact.empty:
        return exact.sort_values(by=["numVotes", "averageRating"], ascending=False).index[0]

    # 2. Contains / starts-with
    contains = df[[t in s for s in TITLES_LOWER]]
    if not contains.empty:
        starts = contains[[TITLES_LOWER[i].startswith(t) for i in contains.index]]
        pool   = starts if not starts.empty else contains
        return pool.sort_values(by=["numVotes", "averageRating"], ascending=False).index[0]

    # 3. Fuzzy fallback
    close = get_close_matches(t, TITLES_LOWER, n=1, cutoff=0.55)
    if close:
        match = df[[s == close[0] for s in TITLES_LOWER]]
        if not match.empty:
            return match.index[0]

    return None


# ─── Recommendation logic ─────────────────────────────────────────────────────
def recommend(title: str) -> list:
    key = title.strip().lower()
    if key in _rec_cache:
        return _rec_cache[key]

    idx = _find_movie_index(title)
    if idx is None:
        return []

    neighbours = NEIGHBOURS.get(idx, [])

    top = [i for i in neighbours if VOTES[i] > 0 and RATINGS[i] > 0][:5]
    if not top:
        top = neighbours[:5]

    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(get_movie_details, df['primaryTitle'].iloc[i]): i
            for i in top
        }
        for future in as_completed(futures):
            i       = futures[future]
            details = future.result()
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

    _rec_cache[key] = results
    return results


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route("/ping")
def ping():
    return "ok", 200

@app.route("/recommend", methods=["POST", "OPTIONS"])
def recommend_api():
    if request.method == "OPTIONS":
        return jsonify({}), 200

    data = request.get_json()
    if not data or "movie" not in data:
        return jsonify({"error": "No movie provided"}), 400

    try:
        results = recommend(data["movie"])
    except Exception as e:
        return jsonify({"error": "Server error", "details": str(e)}), 500

    return jsonify(results)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
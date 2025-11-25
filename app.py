import os
import json
import math
from flask import Flask, jsonify, request
from flask_cors import CORS
from pymongo import MongoClient
from openai import OpenAI

# =====================================================
# Load .env
# =====================================================
try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

# =====================================================
# MongoDB
# =====================================================
MONGO_URL = os.getenv("MONGO_URL")
if not MONGO_URL:
    raise RuntimeError("❌ Missing MONGO_URL in .env")

mongo_client = MongoClient(MONGO_URL)
db = mongo_client["wordcrack"]
words_col = db["words"]

# =====================================================
# OpenAI
# =====================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# Flask
app = Flask(__name__)
CORS(app)

# =====================================================
# Helpers
# =====================================================
def doc_to_dict(doc):
    return {
        "id": str(doc.get("_id")),
        "word": doc.get("word"),
        "chinese": doc.get("chinese"),
        "part_of_speech": doc.get("part_of_speech"),
        "level": doc.get("level"),
    }

def fix_doc(d):
    chinese = d.get("chinese")
    if chinese is None:
        d["chinese"] = ""
    elif isinstance(chinese, float) and math.isnan(chinese):
        d["chinese"] = ""
    return d

# =====================================================
# Health Check
# =====================================================
@app.route("/api/health")
def health():
    try:
        mongo_client.admin.command("ping")
        return jsonify({"ok": True})
    except Exception:
        return jsonify({"ok": False})

# =====================================================
# Get All Words
# =====================================================
@app.route("/api/words")
def get_words():
    try:
        cursor = words_col.find({}, {"embedding": 0}).sort("word", 1)
        return jsonify([fix_doc(doc_to_dict(x)) for x in cursor])
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =====================================================
# Search by First Letter
# =====================================================
@app.route("/api/words/by_letter/<letter>")
def by_letter(letter):
    regex = {"$regex": f"^{letter}", "$options": "i"}
    cursor = words_col.find({"word": regex}, {"embedding": 0}).sort("word", 1)
    return jsonify([fix_doc(doc_to_dict(x)) for x in cursor])

# =====================================================
# General Search
# =====================================================
@app.route("/api/search")
def search():
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify([])

    regex = {"$regex": q, "$options": "i"}

    cursor = words_col.find(
        {"$or": [
            {"word": regex},
            {"chinese": regex},
            {"part_of_speech": regex}
        ]},
        {"embedding": 0}
    ).sort("word", 1)

    return jsonify([fix_doc(doc_to_dict(x)) for x in cursor])

# =====================================================
# Words by Level
# =====================================================
@app.route("/api/words/level/<int:lvl>")
def words_by_level(lvl):
    cursor = words_col.find({"level": lvl}, {"embedding": 0}).sort("word", 1)
    return jsonify([fix_doc(doc_to_dict(x)) for x in cursor])

# =====================================================
# Vector Search
# =====================================================
@app.route("/api/words/similar_db", methods=["POST"])
def similar_db():
    payload = request.get_json(force=True)
    word = payload.get("word", "").strip()
    top_k = payload.get("top_k", 5)

    if not word:
        return jsonify([])

    base = words_col.find_one({"word": word})
    if not base or "embedding" not in base:
        return jsonify([])

    query_vec = base["embedding"]

    pipeline = [
        {
            "$vectorSearch": {
                "index": "embedding_index",
                "path": "embedding",
                "queryVector": query_vec,
                "numCandidates": 200,
                "limit": top_k + 1
            }
        },
        {
            "$project": {
                "word": 1,
                "chinese": 1,
                "part_of_speech": 1,
                "level": 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        }
    ]

    docs = list(words_col.aggregate(pipeline))
    results = []

    for d in docs:
        if d.get("word") == word:
            continue

        results.append({
            "word": d.get("word"),
            "chinese": d.get("chinese", ""),
            "part_of_speech": d.get("part_of_speech", ""),
            "level": d.get("level", ""),
            "score": d.get("score")
        })

        if len(results) >= top_k:
            break

    return jsonify(results)

# =====================================================
# Sentence API
# =====================================================
@app.route("/api/words/sentence", methods=["POST"])
def sentence():
    data = request.get_json(force=True)
    word = data.get("word", "").strip()

    if not OPENAI_API_KEY or not word:
        return jsonify({
            "sentence": f"I saw the word '{word}' today.",
            "translation": f"我今天看到了「{word}」。"
        })

    prompt = f"""
你是一位英文老師。請為單字「{word}」寫一個自然、生活化的英文例句（至少 10 字）。
務必只輸出 JSON。

範例格式：

{{
  "sentence": "I have the ability to solve this problem with patience.",
  "translation": "我有能力以耐心解決這個問題。"
}}
"""

    try:
        res = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )

        raw = res.choices[0].message.content.strip()
        response_json = json.loads(raw)
        return jsonify(response_json)

    except Exception as e:
        return jsonify({
            "sentence": f"I used the word '{word}' today.",
            "translation": f"我今天用了「{word}」。",
            "error": str(e)
        })

# =====================================================
# RUN SERVER — Railway Friendly
# =====================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)
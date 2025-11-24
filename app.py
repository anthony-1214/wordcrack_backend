import os
import json
from flask import Flask, jsonify, request
from flask_cors import CORS
from pymongo import MongoClient
from openai import OpenAI

# =====================================================
# 載入 .env
# =====================================================
try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

# =====================================================
# MongoDB 連線
# =====================================================
MONGO_URL = os.getenv("MONGO_URL")
if not MONGO_URL:
    raise RuntimeError("❌ 請在 .env 設定 MONGO_URL")

mongo_client = MongoClient(MONGO_URL)
db = mongo_client["wordcrack"]
words_col = db["words"]

# =====================================================
# OpenAI
# =====================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

app = Flask(__name__)
CORS(app)

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
# 將 Mongo 文件轉成前端格式
# =====================================================
def doc_to_dict(doc):
    return {
        "id": str(doc.get("_id")),
        "word": doc.get("word"),
        "chinese": doc.get("chinese"),
        "part_of_speech": doc.get("part_of_speech"),
        "level": doc.get("level"),
    }


# =====================================================
# ⭐ 取得全部單字
# =====================================================
@app.route("/api/words")
def get_words():
    cursor = words_col.find({}, {"embedding": 0}).sort("word", 1)
    return jsonify([doc_to_dict(x) for x in cursor])


# =====================================================
# ⭐ 搜尋單字
# =====================================================
@app.route("/api/search")
def search():
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify([])

    regex = {"$regex": q, "$options": "i"}

    cursor = words_col.find(
        {
            "$or": [
                {"word": regex},
                {"chinese": regex},
                {"part_of_speech": regex},
            ]
        },
        {"embedding": 0}
    ).sort("word", 1)

    return jsonify([doc_to_dict(x) for x in cursor])


# =====================================================
# ⭐ Vector Search
# =====================================================
@app.route("/api/words/similar_db", methods=["POST"])
def similar_db():
    payload = request.get_json(force=True)
    word = payload.get("word", "").strip()
    top_k = payload.get("top_k", 5)

    if not word:
        return jsonify([])

    # 找該字的 embedding
    base = words_col.find_one({"word": word})
    if not base or "embedding" not in base:
        return jsonify([])

    query_vec = base["embedding"]

    # 使用 MongoDB Atlas Vector Search
    pipeline = [
        {
            "$vectorSearch": {
                "index": "embedding_index",     # 你 Atlas 上的索引名稱
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
# ⭐ 例句生成
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
    為單字「{word}」寫一個自然、生活化的英文例句（至少 10 字）。
    回傳 JSON 格式：
    {{
        "sentence": "...",
        "translation": "..."
    }}
    """

    res = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )

    raw = res.choices[0].message.content.strip()

    try:
        return jsonify(json.loads(raw))
    except:
        return jsonify({
            "sentence": f"I used the word '{word}' today.",
            "translation": f"我今天用了「{word}」。"
        })


# =====================================================
# RUN
# =====================================================
if __name__ == "__main__":
    app.run(port=5001, debug=True)
import os
import json
import math
from flask import Flask, jsonify, request
from flask_cors import CORS
from pymongo import MongoClient
from openai import OpenAI

# =========================================
# Load .env
# =========================================
try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

# =========================================
# MongoDB 連線
# =========================================
MONGO_URL = os.getenv("MONGO_URL")
if not MONGO_URL:
    raise RuntimeError("❌ Missing MONGO_URL in environment")

mongo_client = MongoClient(MONGO_URL)
db = mongo_client["wordcrack"]
words_col = db["words"]

# =========================================
# OpenAI
# =========================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

app = Flask(__name__)
CORS(app)

# =========================================
# Helper
# =========================================
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

# =========================================
# Health Check
# =========================================
@app.route("/api/health")
def health():
    try:
        mongo_client.admin.command("ping")
        return jsonify({"ok": True})
    except:
        return jsonify({"ok": False})

# =========================================
# 取得全部單字
# =========================================
@app.route("/api/words")
def get_words():
    try:
        cursor = words_col.find({}, {"embedding": 0}).sort("word", 1)
        return jsonify([fix_doc(doc_to_dict(x)) for x in cursor])
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =========================================
# A–Z Filter
# =========================================
@app.route("/api/words/by_letter/<letter>")
def by_letter(letter):
    regex = {"$regex": f"^{letter}", "$options": "i"}
    cursor = words_col.find({"word": regex}, {"embedding": 0}).sort("word", 1)
    return jsonify([fix_doc(doc_to_dict(x)) for x in cursor])

# =========================================
# 搜尋
# =========================================
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

# =========================================
# Level Filter
# =========================================
@app.route("/api/words/level/<int:lvl>")
def words_by_level(lvl):
    cursor = words_col.find({"level": lvl}, {"embedding": 0}).sort("word", 1)
    return jsonify([fix_doc(doc_to_dict(x)) for x in cursor])

# =========================================
# 向量相似度（同義字）
# =========================================
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
            "score": d.get("score"),
        })

        if len(results) >= top_k:
            break

    return jsonify(results)

# =========================================
# 🔥 AI 例句（強化 JSON Parser，不會再報錯）
# =========================================
@app.route("/api/words/sentence", methods=["POST"])
def sentence():
    data = request.get_json(force=True)
    word = data.get("word", "").strip()

    # 沒 key 或沒字 → fallback
    if not OPENAI_API_KEY or not word:
        return jsonify({
            "sentence": f"I saw the word '{word}' today.",
            "translation": f"我今天看到了「{word}」。"
        })

    # 💬 Prompt：強制只能輸出 JSON
    prompt = f"""
請嚴格輸出以下 JSON 格式（不要加任何多餘文字）：

{{
  "sentence": "英文例句（至少 10 個字）",
  "translation": "中文翻譯"
}}

單字：{word}
    """

    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        raw = res.choices[0].message.content.strip()

        # -------- 修正 OpenAI 可能輸出 code block --------
        raw = raw.replace("```json", "").replace("```", "").strip()

        # -------- 自動擷取 JSON 區塊 --------
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1:
            raw = raw[start:end+1]

        # -------- 解析 JSON --------
        response_json = json.loads(raw)
        return jsonify(response_json)

    except Exception as e:
        # fallback（不會壞）
        return jsonify({
            "sentence": f"I used the word '{word}' today.",
            "translation": f"我今天用了「{word}」。",
            "error": str(e)
        })

# =========================================
# RUN (for Render)
# =========================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)
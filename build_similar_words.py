"""
build_similar_words.py
從 word_embeddings + words 產生「每個單字的相似字表」寫入 MySQL 的 similar_words 資料表。

前置：
1. 你已經跑過 embed_words.py，資料表 word_embeddings 已經有資料。
2. backend/.env 內要有 MYSQL_URL，比如：
   MYSQL_URL=mysql://root:xxxx@turntable.proxy.rlwy.net:24042/railway

3. 需要套件：
   pip install pymysql numpy python-dotenv
"""

import os
import json
import traceback
from urllib.parse import urlparse

import numpy as np
import pymysql
from dotenv import load_dotenv

# -----------------------------
# 讀取 .env
# -----------------------------
load_dotenv()
MYSQL_URL = os.getenv("MYSQL_URL", "")

if not MYSQL_URL:
    raise RuntimeError("❌ 缺少 MYSQL_URL，請在 .env 設定，例如：mysql://root:...@host:port/railway")

# -----------------------------
# 建立 DB 連線
# -----------------------------
def get_db():
    url = urlparse(MYSQL_URL)
    return pymysql.connect(
        host=url.hostname,
        user=url.username,
        password=url.password,
        database=url.path[1:],
        port=url.port,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=False,
    )

db = get_db()
print("✅ 已連線 MySQL")

# -----------------------------
# 建立 / 確認 similar_words 資料表
# -----------------------------
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS similar_words (
  id INT AUTO_INCREMENT PRIMARY KEY,
  base_word VARCHAR(255) NOT NULL,
  similar_word VARCHAR(255) NOT NULL,
  score FLOAT NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  UNIQUE KEY uq_base_sim (base_word, similar_word)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

with db.cursor() as cursor:
    cursor.execute(CREATE_TABLE_SQL)
db.commit()
print("✅ 已確認建立資料表：similar_words")

# -----------------------------
# 讀取所有 word + embedding
# -----------------------------
print("📥 讀取 word_embeddings...")

with db.cursor() as cursor:
    cursor.execute("""
        SELECT w.id, w.word, w.chinese, e.embedding
        FROM words w
        JOIN word_embeddings e ON w.id = e.word_id
        ORDER BY w.id
    """)
    rows = cursor.fetchall()

if not rows:
    db.close()
    raise SystemExit("❌ word_embeddings 裡沒有資料，請先跑 embed_words.py")

# 準備 numpy 陣列
words = []
chinese = []
emb_list = []

for r in rows:
    words.append(r["word"])
    chinese.append(r["chinese"])
    vec = np.array(json.loads(r["embedding"]), dtype="float32")
    emb_list.append(vec)

emb_matrix = np.vstack(emb_list)  # (N, D)
N, D = emb_matrix.shape
print(f"🧮 共載入 {N} 個單字，向量維度 {D}")

# -----------------------------
# 計算 cosine similarity matrix
# -----------------------------
print("⚙️ 正規化向量...")
norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
emb_norm = emb_matrix / norms

print("🔢 計算 N x N 相似度矩陣（可能需要一點時間）...")
# (N, D) @ (D, N) = (N, N)
sim_matrix = np.dot(emb_norm, emb_norm.T)

# 自己跟自己設成 -inf，避免被選進相似字
np.fill_diagonal(sim_matrix, -1.0)

TOP_K = 5  # 每個字取幾個相似字

# -----------------------------
# 寫入 similar_words
# -----------------------------
print("📝 清空舊的 similar_words 資料（可視需求保留）...")
with db.cursor() as cursor:
    cursor.execute("TRUNCATE TABLE similar_words;")
db.commit()

print(f"🚀 開始為每個單字寫入前 {TOP_K} 個相似字...")

batch_values = []
BATCH_SIZE = 1000

for i in range(N):
    sims = sim_matrix[i]
    # 取前 TOP_K 大的 index
    if TOP_K >= N:
        top_idx = np.argsort(-sims)
    else:
        # argpartition 比 argsort 快
        top_idx = np.argpartition(-sims, TOP_K)[:TOP_K]
        top_idx = top_idx[np.argsort(-sims[top_idx])]

    base_word = words[i]

    for j in top_idx:
        sim_word = words[j]
        score = float(sims[j])
        batch_values.append((base_word, sim_word, score))

    # 批次寫入
    if len(batch_values) >= BATCH_SIZE:
        with db.cursor() as cursor:
            cursor.executemany(
                """
                INSERT INTO similar_words (base_word, similar_word, score)
                VALUES (%s, %s, %s)
                ON DUPLICATE KEY UPDATE
                  score = VALUES(score)
                """,
                batch_values,
            )
        db.commit()
        print(f"✅ 已寫入 {len(batch_values)} 筆（中途累計），目前處理到第 {i+1} / {N} 個單字")
        batch_values.clear()

# 寫入剩餘的
if batch_values:
    with db.cursor() as cursor:
        cursor.executemany(
            """
            INSERT INTO similar_words (base_word, similar_word, score)
            VALUES (%s, %s, %s)
            ON DUPLICATE KEY UPDATE
              score = VALUES(score)
            """,
            batch_values,
        )
    db.commit()
    print(f"✅ 最後補寫 {len(batch_values)} 筆")

db.close()
print("🎉 全部完成！similar_words 已生成全表。")
"""Day 7 — một chỗ duy nhất cho đường dẫn, model và tham số cố định.

Mọi script import từ đây để dev/test chạy cùng một cấu hình.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# ---- Day 6: vector store + embedding (KHÔNG đổi; query embedding phải khớp index)
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://day6:day6_local_only@127.0.0.1:55433/day6_lab")
DAY6_INDEX = "B"                       # Day 6 chọn C = index B + filter region
EMBED_MODEL = "intfloat/multilingual-e5-base"
EMBED_REVISION = "d128750597153bb5987e10b1c3493a34e5a4502a"
EMBED_DIM = 768
QUERY_PREFIX = "query: "               # E5: câu hỏi dùng "query: ", chunk dùng "passage: "
SOURCE_ID = "iucn-wildcats-1996"

# ---- Dữ liệu Day 7
CHUNKS_PATH = ROOT / "data" / "corpus" / "chunks.json"
QUESTIONS_PATH = ROOT / "data" / "benchmark" / "questions.json"   # câu có đáp án (có gold)
CASES_PATH = ROOT / "data" / "benchmark" / "cases.json"           # toàn bộ câu + answerable
LOCK_PATH = ROOT / "data" / "benchmark" / "LOCK.json"
RESULTS = ROOT / "results"

# ---- Agent model: Gemini API (key trong .env, không commit). So sánh 2 model trên dev.
AGENT_MODELS = ("gemini-3.5-flash", "gemini-3.5-flash-lite")
AGENT_TEMPERATURE = 0.0
AGENT_SEED = 7

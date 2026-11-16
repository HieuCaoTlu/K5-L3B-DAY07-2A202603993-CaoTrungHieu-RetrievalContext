"""Bước 1: xuất chunk index B từ pgvector Day 6 sang schema Day 7 (data/corpus/chunks.json).

Chạy từ thư mục gốc:  python src/export_day6.py
Chỉ đọc DB, không sửa index. Ghi thêm data/corpus/day6_connection.json (phiên bản, số chiều, số chunk).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rag import config  # noqa: E402
from rag.day6 import Day6Store  # noqa: E402


def main() -> None:
    store = Day6Store()
    info = store.info()
    rows = store.all_chunks()
    chunks = []
    for r in rows:
        chunks.append({
            "id": r["chunk_id"], "text": r["text"], "source_id": r["document_id"], "source": r["source"],
            "page": r["page"],
            "metadata": {"region": r["region"], "species": r["species"], "section": r["section"],
                         "char_start": r["char_start"], "char_end": r["char_end"], "day6_index": config.DAY6_INDEX},
        })
    if len({c["id"] for c in chunks}) != len(chunks):
        raise SystemExit("chunk id trùng")
    config.CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.CHUNKS_PATH.write_text(json.dumps(chunks, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    info.update({"embedding_revision": config.EMBED_REVISION, "query_prefix": config.QUERY_PREFIX,
                 "pages": len({c["page"] for c in chunks}), "exported_chunks": len(chunks),
                 "search": "exact cosine, sequential scan (no ANN)", "database_url": "local docker (see .env.example)"})
    (config.CHUNKS_PATH.parent / "day6_connection.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Kết nối tới vector store + embedding của Day 6 (PostgreSQL 16 + pgvector 0.8.7, E5-base).

- Query embedding dùng đúng model/revision/prefix của lúc index -> tương thích vector.
- Không fallback sang vector giả: model hoặc DB lỗi thì raise để trace ghi lỗi thật.
- Cột filter lấy từ danh sách cho phép; giá trị truyền dạng tham số SQL.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import psycopg

from . import config

FILTER_COLUMNS = {"region"}
META_COLUMNS = ["document_id", "source", "page", "region", "species", "section", "char_start", "char_end"]


def _vec(v: list[float]) -> str:
    return "[" + ",".join(f"{x:.7f}" for x in v) + "]"


class E5QueryEmbedder:
    def __init__(self) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(config.EMBED_MODEL, device=device, revision=config.EMBED_REVISION)
        self.dimensions = int(self.model.get_sentence_embedding_dimension())
        if self.dimensions != config.EMBED_DIM:
            raise RuntimeError(f"embedding dim {self.dimensions} != index dim {config.EMBED_DIM}")
        self.device = device

    def embed_query(self, text: str) -> list[float]:
        v = self.model.encode([config.QUERY_PREFIX + text], normalize_embeddings=True,
                              convert_to_numpy=True, show_progress_bar=False)[0]
        return v.tolist()


@lru_cache(maxsize=1)
def embedder() -> E5QueryEmbedder:
    return E5QueryEmbedder()


class Day6Store:
    def __init__(self, url: str = config.DATABASE_URL, index: str = config.DAY6_INDEX) -> None:
        self.conn = psycopg.connect(url, connect_timeout=5, autocommit=True)
        self.index = index

    def info(self) -> dict[str, Any]:
        pg, ext = self.conn.execute("SELECT current_setting('server_version'), extversion FROM pg_extension "
                                    "WHERE extname='vector'").fetchone()
        n, model = self.conn.execute("SELECT count(*), min(embedding_model) FROM chunks WHERE index_name=%s",
                                     (self.index,)).fetchone()
        dim = self.conn.execute("SELECT atttypmod FROM pg_attribute WHERE attrelid='chunks'::regclass "
                                "AND attname='embedding'").fetchone()[0]
        return {"postgres": pg, "pgvector": ext, "index": self.index, "chunks": n, "embedding_model": model,
                "vector_dim": dim}

    def all_chunks(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            f"SELECT chunk_id, text, {', '.join(META_COLUMNS)} FROM chunks WHERE index_name=%s "
            "ORDER BY page, char_start", (self.index,)).fetchall()
        return [{"chunk_id": r[0], "text": r[1], **dict(zip(META_COLUMNS, r[2:]))} for r in rows]

    def dense_search(self, query_vector: list[float], top_k: int,
                     filters: dict[str, Any] | None = None) -> list[tuple[str, float]]:
        """Exact cosine (seq scan, không ANN) -> [(chunk_id, score)], score = 1 - cosine distance."""
        where, params = ["index_name = %s"], [self.index]
        for col, val in (filters or {}).items():
            if col not in FILTER_COLUMNS:
                raise ValueError(f"filter column not allowed: {col}")
            where.append(f"{col} = %s")
            params.append(val)
        qv = _vec(query_vector)
        rows = self.conn.execute(
            f"SELECT chunk_id, 1 - (embedding <=> %s::vector) FROM chunks WHERE {' AND '.join(where)} "
            "ORDER BY embedding <=> %s::vector, chunk_id LIMIT %s", [qv, *params, qv, top_k]).fetchall()
        return [(r[0], float(r[1])) for r in rows]

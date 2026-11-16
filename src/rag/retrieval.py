"""Bước 2: các phương pháp retrieval, cùng một interface.

    retriever.search(query, filters=None, top_k=10) -> list[dict]

Mỗi kết quả: id, text, source_id, source, page, metadata, score, rank, strategy (+ điểm thành phần).

| Tên          | Cách tìm                                                     | Dùng index         |
|--------------|--------------------------------------------------------------|--------------------|
| dense        | E5 query embedding -> pgvector cosine chính xác (Day 6)       | pgvector index B   |
| bm25         | Okapi BM25 (k1=1.5, b=0.75), tách từ tiếng Anh + bỏ stopword  | chunks.json (=B)   |
| hybrid       | RRF (k=60) trên top-50 dense + top-50 bm25                    | cả hai             |
| *+rerank     | lấy top-30 candidate rồi xếp lại bằng cross-encoder MiniLM-L12| cross-encoder      |

Filter: chỉ cột `region`, chỉ khi câu hỏi mang filter do người dùng cung cấp (giống Day 6 cấu hình C).
BM25 lọc trong Python trên cùng metadata; dense lọc bằng WHERE trong SQL.
Lỗi DB/model được raise lên, không thay bằng kết quả rỗng.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections import Counter
from functools import lru_cache
from typing import Any

from . import config

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"
RERANK_REVISION = "7b0235231ca2674cb8ca8f022859a6eba2b1c968"

STOPWORDS = set("""a an the and or of to in on at by for with from as is are was were be been being this that these those
it its it's into than then there their they them which who whom whose what when where why how do does did not no
but if so such can could may might will would should has have had also only about over under between within out up
down very more most other some any each per i you we he she his her our your one""".split())


def check_lock() -> None:
    """Dừng nếu benchmark hoặc corpus bị sửa sau khi khoá."""
    lock = json.loads(config.LOCK_PATH.read_text(encoding="utf-8"))
    for key, path in (("questions_sha256", config.QUESTIONS_PATH), ("cases_sha256", config.CASES_PATH),
                      ("chunks_sha256", config.CHUNKS_PATH)):
        if hashlib.sha256(path.read_bytes()).hexdigest() != lock[key]:
            raise SystemExit(f"{path.name} đã thay đổi sau khi khoá benchmark -> từ chối chạy.")


def normalize(text: str) -> str:
    # Ghép từ bị ngắt dòng OCR: "ani-\n\nmals" -> "animals"
    return re.sub(r"(\w)-\s*\n+\s*(\w)", r"\1\2", text)


def tokenize(text: str) -> list[str]:
    toks = re.findall(r"[a-z0-9]+(?:[.,][0-9]+)*", normalize(text).lower())
    out = []
    for t in toks:
        if t in STOPWORDS or len(t) == 1 and not t.isdigit():
            continue
        # stem rất nhẹ: số nhiều tiếng Anh (cats->cat, leopards->leopard, species giữ nguyên)
        if len(t) > 4 and t.endswith("s") and not t.endswith(("ss", "us", "is")):
            t = t[:-1]
        out.append(t)
    return out


@lru_cache(maxsize=1)
def load_chunks() -> tuple[dict[str, Any], ...]:
    return tuple(json.loads(config.CHUNKS_PATH.read_text(encoding="utf-8")))


def chunk_by_id() -> dict[str, dict[str, Any]]:
    return {c["id"]: c for c in load_chunks()}


def _match(chunk: dict, filters: dict | None) -> bool:
    return all(chunk["metadata"].get(k) == v for k, v in (filters or {}).items())


def _result(chunk: dict, score: float, rank: int, strategy: str, **extra) -> dict[str, Any]:
    return {"id": chunk["id"], "text": chunk["text"], "source_id": chunk["source_id"], "source": chunk["source"],
            "page": chunk["page"], "metadata": chunk["metadata"], "score": round(float(score), 6), "rank": rank,
            "strategy": strategy, **extra}


class BM25Retriever:
    name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        t0 = time.perf_counter()
        self.chunks = list(load_chunks())
        self.k1, self.b = k1, b
        self.tf = [Counter(tokenize(c["text"])) for c in self.chunks]
        self.len = [sum(tf.values()) for tf in self.tf]
        self.avgdl = sum(self.len) / len(self.len)
        df = Counter(t for tf in self.tf for t in tf)
        n = len(self.chunks)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
        self.index_seconds = time.perf_counter() - t0

    def scores(self, query: str, filters: dict | None = None) -> list[tuple[int, float]]:
        q = [t for t in tokenize(query) if t in self.idf]
        out = []
        for i, (c, tf, dl) in enumerate(zip(self.chunks, self.tf, self.len)):
            if not _match(c, filters):
                continue
            s = 0.0
            for t in q:
                f = tf.get(t, 0)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            if s > 0:
                out.append((i, s))
        out.sort(key=lambda x: (-x[1], self.chunks[x[0]]["id"]))
        return out

    def search(self, query: str, filters: dict | None = None, top_k: int = 10) -> list[dict[str, Any]]:
        return [_result(self.chunks[i], s, r, self.name) for r, (i, s) in enumerate(self.scores(query, filters)[:top_k], 1)]


class DenseRetriever:
    name = "dense"

    def __init__(self) -> None:
        from .day6 import Day6Store, embedder
        self.store = Day6Store()
        self.embedder = embedder()
        self.by_id = chunk_by_id()

    def search(self, query: str, filters: dict | None = None, top_k: int = 10) -> list[dict[str, Any]]:
        qv = self.embedder.embed_query(query)
        rows = self.store.dense_search(qv, top_k, filters)
        missing = [cid for cid, _ in rows if cid not in self.by_id]
        if missing:
            raise RuntimeError(f"pgvector trả chunk không có trong chunks.json (index đã đổi?): {missing[:3]}")
        return [_result(self.by_id[cid], s, r, self.name) for r, (cid, s) in enumerate(rows, 1)]


class HybridRRF:
    """Reciprocal Rank Fusion: score = sum 1/(k + rank). Không cộng điểm thô (thang đo khác nhau)."""
    name = "hybrid"

    def __init__(self, dense: DenseRetriever, bm25: BM25Retriever, k: int = 60, depth: int = 50) -> None:
        self.dense, self.bm25, self.k, self.depth = dense, bm25, k, depth
        self.by_id = chunk_by_id()

    def search(self, query: str, filters: dict | None = None, top_k: int = 10) -> list[dict[str, Any]]:
        d = self.dense.search(query, filters, self.depth)
        b = self.bm25.search(query, filters, self.depth)
        fused: dict[str, dict] = {}
        for lst, key in ((d, "dense"), (b, "bm25")):
            for r in lst:
                e = fused.setdefault(r["id"], {"rrf": 0.0})
                e["rrf"] += 1.0 / (self.k + r["rank"])
                e[f"{key}_rank"], e[f"{key}_score"] = r["rank"], r["score"]
        order = sorted(fused.items(), key=lambda kv: (-kv[1]["rrf"], kv[0]))[:top_k]
        return [_result(self.by_id[cid], e["rrf"], r, self.name,
                        **{k: v for k, v in e.items() if k != "rrf"}) for r, (cid, e) in enumerate(order, 1)]


@lru_cache(maxsize=1)
def cross_encoder():
    import torch
    from sentence_transformers import CrossEncoder
    return CrossEncoder(RERANK_MODEL, revision=RERANK_REVISION, max_length=512,
                        device="cuda" if torch.cuda.is_available() else "cpu")


class Reranked:
    """Lấy `depth` candidate từ base rồi xếp lại bằng cross-encoder (query, chunk)."""

    def __init__(self, base, depth: int = 30) -> None:
        self.base, self.depth = base, depth
        self.name = f"{base.name}+rerank"
        self.model = cross_encoder()

    def search(self, query: str, filters: dict | None = None, top_k: int = 10) -> list[dict[str, Any]]:
        cands = self.base.search(query, filters, self.depth)
        if not cands:
            return []
        scores = self.model.predict([(query, normalize(c["text"])) for c in cands], batch_size=16,
                                    show_progress_bar=False)
        order = sorted(zip(cands, scores), key=lambda x: (-float(x[1]), x[0]["id"]))[:top_k]
        out = []
        for r, (c, s) in enumerate(order, 1):
            extra = {k: v for k, v in c.items() if k.endswith(("_rank", "_score"))}
            out.append(_result(c, float(s), r, self.name, base_rank=c["rank"], base_score=c["score"], **extra))
        return out


def build(name: str, *, _cache: dict = {}) -> Any:
    """Factory dùng chung (giữ 1 bản model/DB connection)."""
    if name in _cache:
        return _cache[name]
    if name == "bm25":
        r = BM25Retriever()
    elif name == "dense":
        r = DenseRetriever()
    elif name == "hybrid":
        r = HybridRRF(build("dense"), build("bm25"))
    elif name.endswith("+rerank"):
        r = Reranked(build(name[: -len("+rerank")]))
    else:
        raise ValueError(f"retriever không hỗ trợ: {name}")
    _cache[name] = r
    return r

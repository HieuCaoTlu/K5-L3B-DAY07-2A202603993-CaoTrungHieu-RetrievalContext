"""Chọn context gửi cho model.

Đơn vị budget: KÝ TỰ của toàn bộ block (gồm header nguồn), ~4 ký tự ≈ 1 token tiếng Anh.

Chiến lược:
- "rank":      lấy theo thứ tự rank, bỏ chunk trùng ID, chunk không vừa budget thì bỏ qua (thử chunk sau).
- "neighbors": như "rank" nhưng mỗi chunk được chọn kéo thêm chunk liền kề (cùng trang, trước/sau)
               nếu còn budget -> giữ câu bị cắt ở ranh giới chunk (lỗi d07/d19 kiểu Day 6).
- "trim":      như "rank" nhưng chunk quá dài bị cắt quanh câu khớp query nhiều nhất (nén context).

Header mỗi block giữ id + trang + vùng + loài để model trích dẫn và kiểm chứng.
Nội dung tài liệu được bọc như DỮ LIỆU, không phải chỉ dẫn (xem prompt agent).
"""
from __future__ import annotations

import re
from typing import Any

from .retrieval import chunk_by_id, load_chunks, normalize, tokenize


def _block(c: dict[str, Any], text: str | None = None, header: str = "full") -> str:
    """header="full": page + region + species (species từ Day 6, SAI ở trang chuyển bài);
    header="no_species": chỉ page + region (2 trường Day 6 kiểm chứng đúng 90/90)."""
    m = c["metadata"]
    meta = f"page {c['page']}, {m['region']}" + (f", section: {m['species']}" if header == "full" else "")
    return f"[{c['id']}] ({meta})\n{normalize(text or c['text']).strip()}"


_NEIGH: dict[str, list[str]] = {}


def neighbors(cid: str) -> list[str]:
    if not _NEIGH:
        chunks = list(load_chunks())   # đã sắp theo page, char_start
        for i, c in enumerate(chunks):
            _NEIGH[c["id"]] = [chunks[j]["id"] for j in (i - 1, i + 1)
                               if 0 <= j < len(chunks) and chunks[j]["page"] == c["page"]]
    return _NEIGH.get(cid, [])


def _trim(c: dict[str, Any], query: str, max_chars: int) -> str:
    text = normalize(c["text"])
    if len(text) <= max_chars:
        return text
    sents = re.split(r"(?<=[.!?])\s+", text)
    q = set(tokenize(query))
    best = max(range(len(sents)), key=lambda i: len(q & set(tokenize(sents[i]))))
    lo = hi = best
    out = sents[best]
    while True:
        grown = False
        for j in (lo - 1, hi + 1):
            if 0 <= j < len(sents) and len(out) + len(sents[j]) + 1 <= max_chars:
                out = (sents[j] + " " + out) if j < lo else (out + " " + sents[j])
                lo, hi = min(lo, j), max(hi, j)
                grown = True
        if not grown:
            return out


def pack(results: list[dict[str, Any]], budget: int = 3000, mode: str = "rank", query: str = "",
         max_chunks: int | None = None) -> tuple[str, list[str]]:
    by_id = chunk_by_id()
    blocks: list[str] = []
    ids: list[str] = []
    used = 0

    def add(c: dict[str, Any], text: str | None = None) -> bool:
        nonlocal used
        if c["id"] in ids:
            return False
        b = _block(c, text)
        cost = len(b) + (2 if blocks else 0)
        if used + cost > budget:
            return False
        blocks.append(b)
        ids.append(c["id"])
        used += cost
        return True

    for r in results:
        if max_chunks is not None and len(ids) >= max_chunks:
            break
        if mode == "trim":
            add(r, _trim(r, query, 700))
        else:
            add(r)
            if mode == "neighbors" and r["id"] in ids:
                for nid in neighbors(r["id"]):
                    add(by_id[nid])
    return "\n\n".join(blocks), ids

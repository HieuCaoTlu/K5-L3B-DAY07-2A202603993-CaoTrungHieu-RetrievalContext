"""Bước 2/4: đánh giá retrieval-only trên một split (query gốc, filter do người dùng cung cấp).

    python src/eval_retrieval.py --split dev --methods dense bm25 hybrid dense+rerank hybrid+rerank

Metrics (câu có đáp án): Hit@1, Evidence Hit@3, MRR@10, Hit@10, CER (context top-3, budget 3000 ký tự).
Câu thiếu evidence không vào mẫu số; chỉ ghi điểm top-1 để tham khảo ngưỡng.
Output: results/retrieval/<split>/<method>.json (trace từng câu) + summary.json.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rag import config  # noqa: E402
from rag.context import pack  # noqa: E402
from rag.retrieval import build, check_lock, normalize  # noqa: E402


def is_hit(r: dict, gold: list[dict]) -> bool:
    return any(r["source_id"] == g["source_id"] and r["page"] == g["page"] and g["evidence"] in r["text"] for g in gold)


def context_hit(context: str, gold: list[dict]) -> bool:
    return any(normalize(g["evidence"]) in context for g in gold)


def run(method: str, questions: list[dict], cases: list[dict], top_k: int = 10, ctx_k: int = 3,
        budget: int = 3000) -> dict:
    retriever = build(method)
    traces = []
    for q in questions:
        t0 = time.perf_counter()
        res = retriever.search(q["question"], filters=q.get("filter"), top_k=top_k)
        ms = (time.perf_counter() - t0) * 1000
        ctx, ids = pack(res[:ctx_k], budget=budget, mode="rank", query=q["question"])
        hits = [is_hit(r, q["gold"]) for r in res]
        first = next((i + 1 for i, h in enumerate(hits) if h), None)
        traces.append({"query_id": q["id"], "type": q["type"], "question": q["question"], "filter": q.get("filter"),
                       "ms": round(ms, 1), "first_hit_rank": first,
                       "hit1": first == 1, "hit3": bool(first and first <= 3), "hit10": bool(first),
                       "context_hit": context_hit(ctx, q["gold"]), "context_ids": ids, "context_chars": len(ctx),
                       "gold_pages": [g["page"] for g in q["gold"]],
                       "results": [{k: r[k] for k in r if k != "text"} | {"hit": h, "preview": r["text"][:160]}
                                   for r, h in zip(res, hits)]})
    unans = []
    for c in cases:
        t0 = time.perf_counter()
        res = retriever.search(c["question"], filters=c.get("filter"), top_k=3)
        unans.append({"query_id": c["id"], "ms": round((time.perf_counter() - t0) * 1000, 1),
                      "top1_score": res[0]["score"] if res else None,
                      "top3": [{"id": r["id"], "page": r["page"], "score": r["score"]} for r in res]})
    n = len(traces)

    def frac(key):
        k = sum(t[key] for t in traces)
        return {"n": k, "N": n, "rate": round(k / n, 3)}

    by_type = {}
    for t in traces:
        d = by_type.setdefault(t["type"], {"hit1": 0, "hit3": 0, "N": 0})
        d["hit1"] += t["hit1"]
        d["hit3"] += t["hit3"]
        d["N"] += 1
    summary = {"method": method, "N": n, "hit1": frac("hit1"), "hit3": frac("hit3"), "hit10": frac("hit10"),
               "mrr10": round(sum(1 / t["first_hit_rank"] for t in traces if t["first_hit_rank"]) / n, 3),
               "context_evidence_rate": frac("context_hit"),
               "context": {"k": ctx_k, "budget_chars": budget, "mode": "rank",
                           "avg_chars": round(statistics.mean(t["context_chars"] for t in traces))},
               "median_ms": round(statistics.median(t["ms"] for t in traces), 1),
               "by_type": by_type, "cost_usd": 0.0,
               "misses_at3": [t["query_id"] for t in traces if not t["hit3"]]}
    return {"summary": summary, "traces": traces, "unanswerable_probe": unans}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=("dev", "test"))
    ap.add_argument("--methods", nargs="+", default=["dense", "bm25", "hybrid", "dense+rerank", "hybrid+rerank"])
    args = ap.parse_args()
    check_lock()
    if args.split == "test" and not (config.RESULTS / "selection.json").exists():
        sys.exit("Chưa có results/selection.json -> không chạy test.")
    questions = [q for q in json.loads(config.QUESTIONS_PATH.read_text(encoding="utf-8")) if q["split"] == args.split]
    cases = [c for c in json.loads(config.CASES_PATH.read_text(encoding="utf-8"))
             if c["split"] == args.split and not c["answerable"]]
    out_dir = config.RESULTS / "retrieval" / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for m in args.methods:
        res = run(m, questions, cases)
        (out_dir / f"{m.replace('+', '_')}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n",
                                                             encoding="utf-8")
        s = res["summary"]
        summaries.append(s)
        print(f"{m:15s} Hit@1 {s['hit1']['n']}/{s['N']}  Hit@3 {s['hit3']['n']}/{s['N']}  "
              f"MRR {s['mrr10']}  Hit@10 {s['hit10']['n']}/{s['N']}  CER {s['context_evidence_rate']['n']}/{s['N']}  "
              f"{s['median_ms']} ms  miss@3={s['misses_at3']}")
    (out_dir / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

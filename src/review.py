"""Bước 4: hỗ trợ chấm tay câu trả lời agent + tổng hợp metrics bằng starter/agent_metrics.py.

1) Sinh bảng chấm (đọc người):
   python src/review.py sheet --split dev --config A0
   -> results/agent/<split>/<config>/review_sheet.md : câu hỏi, đáp án chuẩn, evidence gold, câu trả lời,
      nguyên văn từng đoạn được trích dẫn để đối chiếu.

2) Người chấm điền results/agent/<split>/<config>/reviews.json, mỗi câu:
   {"query_id", "answer_grade" (0/1/2, câu có đáp án), "citation_support": [bool per citation],
    "abstention_correct" (bool, câu thiếu evidence), "reason"}

3) Tổng hợp:
   python src/review.py summarize --split dev --config A0
   -> agent_metrics.json (dùng hàm aggregate của starter) + thêm số lượt gọi, thời gian, token, CER.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "starter"))
from rag import quiet  # noqa: E402,F401
from agent_metrics import aggregate  # noqa: E402  (starter)
from rag import config  # noqa: E402
from rag.retrieval import chunk_by_id  # noqa: E402


def _paths(split: str, name: str) -> Path:
    return config.RESULTS / "agent" / split / name


def sheet(split: str, name: str) -> None:
    d = _paths(split, name)
    traces = json.loads((d / "traces.json").read_text(encoding="utf-8"))
    cases = {c["id"]: c for c in json.loads(config.CASES_PATH.read_text(encoding="utf-8"))}
    qs = {q["id"]: q for q in json.loads(config.QUESTIONS_PATH.read_text(encoding="utf-8"))}
    by_id = chunk_by_id()
    lines = [f"# Bảng chấm — {name} / {split}\n"]
    for t in traces:
        c = cases[t["query_id"]]
        lines.append(f"## {t['query_id']} · {c['type']} · status={t['status']} · searches={t['searches']}")
        lines.append(f"- **Q:** {c['question']}")
        if c["answerable"]:
            q = qs[t["query_id"]]
            lines.append(f"- **Expected:** {q['expected_answer']}")
            lines.append(f"- **Gold:** p.{[g['page'] for g in q['gold']]} — “{q['gold'][0]['evidence']}”")
        else:
            lines.append(f"- **Expected:** abstain — {c['absence_reason']}")
        lines.append(f"- **Queries:** {[e['query'] for e in t['events'] if e['event'] == 'search']}")
        lines.append(f"- **Answer:** {t['raw_answer'] or t['answer'] or t['error']}")
        for lab, cid in zip(t["citation_labels"], t["citations"]):
            lines.append(f"  - [{lab}] `{cid}` p.{by_id[cid]['page']}: {by_id[cid]['text'][:700]!r}")
        if t["invalid_citations"]:
            lines.append(f"  - invalid labels: {t['invalid_citations']}")
        lines.append("")
    (d / "review_sheet.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote", d / "review_sheet.md")


def summarize(split: str, name: str) -> dict:
    d = _paths(split, name)
    traces = json.loads((d / "traces.json").read_text(encoding="utf-8"))
    reviews = json.loads((d / "reviews.json").read_text(encoding="utf-8"))
    run = json.loads((d / "run_summary.json").read_text(encoding="utf-8"))
    cases = [c for c in json.loads(config.CASES_PATH.read_text(encoding="utf-8")) if c["split"] == split]
    agg = aggregate(cases, traces, reviews)
    ok = [t for t in traces if t["status"] != "error"]
    agg.update({
        "config": name, "split": split,
        "context_evidence_rate": run["context_evidence_rate"],
        "searches_distribution": run["searches_distribution"],
        "mean_searches": round(statistics.mean(t["searches"] for t in ok), 2) if ok else None,
        "median_seconds": run["median_seconds"], "total_seconds": run["total_seconds"],
        "prompt_tokens": run["prompt_tokens"], "completion_tokens": run["completion_tokens"], "cost_usd": 0.0,
        "invalid_citation_labels": sum(len(t["invalid_citations"]) for t in traces),
    })
    (d / "agent_metrics.json").write_text(json.dumps(agg, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    g = agg["grounded_answer"]
    print(f"{name}/{split}: tool_use {agg['tool_use']['n']}/{agg['tool_use']['N']} | grounded 0/1/2 = "
          f"{g['0']}/{g['1']}/{g['2']} mean {g['mean']:.2f} | citation {agg['citation_validity']['n']}/"
          f"{agg['citation_validity']['N']} | abstain {agg['abstention']['n']}/{agg['abstention']['N']} | "
          f"false-abstain {agg['false_abstention']['n']}/{agg['false_abstention']['N']} | CER "
          f"{run['context_evidence_rate']['n']}/{run['context_evidence_rate']['N']} | median {run['median_seconds']}s")
    return agg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("sheet", "summarize"))
    ap.add_argument("--split", required=True, choices=("dev", "test"))
    ap.add_argument("--config", nargs="+", required=True)
    args = ap.parse_args()
    for name in args.config:
        (sheet if args.action == "sheet" else summarize)(args.split, name)


if __name__ == "__main__":
    main()

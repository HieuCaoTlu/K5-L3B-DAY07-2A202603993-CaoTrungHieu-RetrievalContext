"""Bước 3/4: chạy agent thật trên một split với một cấu hình có tên.

    python src/run_agent.py --split dev --config A0
    python src/run_agent.py --split test --config A0 A2     # chỉ sau khi có results/selection.json

Output: results/agent/<split>/<config>/traces.json  (mỗi câu: tool call, query, kết quả, context,
        câu trả lời, citation, số lượt, token, thời gian, lỗi) + retrieval_in_context.json
        (CER của context agent, tính tự động từ gold SAU khi chạy; gold không vào agent).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rag import quiet  # noqa: E402,F401  (tắt warning, stdout line-buffered)
from rag import config  # noqa: E402
from rag.agent import AgentConfig, run_agent, to_dict  # noqa: E402
from rag.retrieval import check_lock, normalize  # noqa: E402

# Cấu hình có tên. A1/A2 được thêm SAU khi đọc kết quả giai đoạn trước (xem results/PLAN.md + EXPERIMENTS.md).
CONFIGS: dict[str, AgentConfig] = {
    "A0": AgentConfig(name="A0", retriever="dense", top_k=3, budget=3000, context_mode="rank", max_searches=3,
                      model="gemini-3.5-flash"),
}


def live_line(name: str, i: int, n: int, c: dict, run) -> str:
    ans = (run.raw_answer or run.error or "").replace("\n", " ")
    queries = [e["query"] for e in run.events if e["event"] == "search"]
    return (f"[{name}] {i}/{n} {c['id']:4s} {'ANS' if c['answerable'] else 'UNANS'} -> {run.status:8s} "
            f"{run.seconds:5.1f}s searches={run.searches} tokens={run.prompt_tokens}+{run.completion_tokens} "
            f"cit={run.citation_labels}\n      Q: {c['question']}\n      queries: {queries}\n      A: {ans[:300]}")


def load_configs() -> None:
    extra = config.RESULTS / "agent_configs.json"
    if extra.exists():
        for name, kw in json.loads(extra.read_text(encoding="utf-8")).items():
            CONFIGS[name] = AgentConfig(name=name, **kw)


def main() -> None:
    load_configs()
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=("dev", "test"))
    ap.add_argument("--config", nargs="+", required=True)
    ap.add_argument("--only", nargs="*", help="chỉ chạy các query_id này (debug)")
    args = ap.parse_args()
    check_lock()
    if args.split == "test" and not (config.RESULTS / "selection.json").exists():
        sys.exit("Chưa có results/selection.json -> không chạy test.")
    cases = [c for c in json.loads(config.CASES_PATH.read_text(encoding="utf-8")) if c["split"] == args.split]
    gold = {q["id"]: q["gold"] for q in json.loads(config.QUESTIONS_PATH.read_text(encoding="utf-8"))}
    if args.only:
        cases = [c for c in cases if c["id"] in args.only]
    for name in args.config:
        cfg = CONFIGS[name]
        out_dir = config.RESULTS / "agent" / args.split / name
        out_dir.mkdir(parents=True, exist_ok=True)
        traces = []
        live = out_dir / "live.log"            # mở file này trong IDE để xem từng câu ngay khi xong
        live.write_text(f"# {name} / {args.split} — model {cfg.model}, retriever {cfg.retriever}\n", encoding="utf-8")
        for i, c in enumerate(cases, 1):
            # Agent chỉ nhận câu hỏi + filter người dùng; KHÔNG nhận gold/đáp án.
            run = run_agent(c["id"], c["question"], cfg, filters=c.get("filter"))
            traces.append(to_dict(run, cfg))
            line = live_line(name, i, len(cases), c, run)
            print(line, flush=True)
            with live.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
            # ghi trace sau MỖI câu -> dừng giữa chừng vẫn giữ kết quả đã chạy
            (out_dir / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=1) + "\n",
                                                 encoding="utf-8")
        # CER của context agent (đánh giá sau khi chạy)
        cer = []
        for t in traces:
            if t["query_id"] in gold:
                ctx = normalize(t["context"])
                cer.append({"query_id": t["query_id"],
                            "context_hit": any(normalize(g["evidence"]) in ctx for g in gold[t["query_id"]])})
        ok = [t for t in traces if t["status"] != "error"]
        summary = {"config": asdict(cfg), "split": args.split, "cases": len(traces),
                   "errors": len(traces) - len(ok),
                   "status": {s: sum(t["status"] == s for t in traces) for s in ("answered", "abstain", "error")},
                   "context_evidence_rate": {"n": sum(x["context_hit"] for x in cer), "N": len(cer)},
                   "searches_distribution": {str(k): sum(t["searches"] == k for t in traces)
                                             for k in sorted({t["searches"] for t in traces})},
                   "median_seconds": statistics.median(t["seconds"] for t in traces),
                   "total_seconds": round(sum(t["seconds"] for t in traces), 1),
                   "prompt_tokens": sum(t["prompt_tokens"] for t in traces),
                   "completion_tokens": sum(t["completion_tokens"] for t in traces),
                   "cost_usd": 0.0, "context_hits": cer}
        (out_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n",
                                                  encoding="utf-8")
        print(json.dumps({k: v for k, v in summary.items() if k not in ("context_hits", "config")}, ensure_ascii=False))


if __name__ == "__main__":
    main()

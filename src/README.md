# Code Day 7 — Agentic RAG trên kho Wild Cats (IUCN 1996)

## Cấu trúc

| File | Vai trò |
|---|---|
| `rag/config.py` | Đường dẫn, model, tham số cố định |
| `rag/day6.py` | Kết nối pgvector Day 6 + E5 query embedding (cùng model/revision/prefix với index) |
| `rag/retrieval.py` | 4 phương pháp: `dense`, `bm25`, `hybrid` (RRF), `*+rerank` (cross-encoder); `check_lock()` |
| `rag/context.py` | Đóng gói context theo budget ký tự, dedup, header nguồn (`full` / `no_species`) |
| `rag/agent.py` | Agent Gemini tool-calling: tool `search_documents`, điều kiện dừng, kiểm tra citation, trace |
| `export_day6.py` | Xuất 859 chunk index B từ pgvector → `data/corpus/chunks.json` |
| `build_benchmark.py` | Dựng + kiểm chứng + khoá benchmark (`--lock`) |
| `eval_retrieval.py` | Retrieval-only: Hit@1, Evidence Hit@3, MRR@10, Hit@10, CER |
| `run_agent.py` | Chạy agent một cấu hình trên một split; ghi `traces.json`, `live.log`, `run_summary.json` |
| `review.py` | `sheet` sinh bảng chấm tay; `summarize` tổng hợp nhãn bằng `starter/agent_metrics.py` |

## Chạy lại (từ thư mục gốc repo)

```bash
# 0. Môi trường: venv Python 3.12 + requirements.txt; pgvector Day 6 đang chạy (docker compose repo Day 6)
cp .env.example .env            # điền GEMINI_API_KEY

# 1. Dữ liệu + benchmark (đã có sẵn trong repo; chỉ chạy lại nếu cần kiểm chứng)
python src/export_day6.py
python src/build_benchmark.py   # kiểm chứng; --lock chỉ dùng lần đầu

# 2. Retrieval
python src/eval_retrieval.py --split dev
python src/eval_retrieval.py --split test          # cần results/selection.json

# 3-4. Agent (cấu hình A0 trong run_agent.py; A0-lite, A1, A2 trong results/agent_configs.json)
python src/run_agent.py --split dev --config A0 A0-lite A1 A2
python src/review.py sheet --split dev --config A2          # đọc review_sheet.md, ghi nhãn
python src/review.py summarize --split dev --config A0 A0-lite A1 A2
python src/run_agent.py --split test --config A0 A2        # chỉ sau khi chốt selection.json
python src/review.py summarize --split test --config A0 A2

# Test offline (không gọi DB/API). Windows: thêm -X utf8 cho test của starter.
python -X utf8 -m unittest discover -s tests
```

Model và DB lỗi → trace ghi `status: "error"`; không có fallback giả. Mọi script chạy đều kiểm tra hash `data/benchmark/LOCK.json`.

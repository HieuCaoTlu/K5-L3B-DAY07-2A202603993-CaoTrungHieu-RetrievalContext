# Kế hoạch thí nghiệm & tiêu chí chọn (ghi TRƯỚC khi chạy dev)

Ghi lúc 2026-11-16 ~09:20 (+07), sau khi khoá benchmark (`data/benchmark/LOCK.json`, 09:13:45),
trước mọi lần chạy retrieval/agent Day 7.

## Điều kiện so sánh cố định

- Corpus: 859 chunk index B Day 6 (`data/corpus/chunks.json`, hash trong LOCK).
- Câu hỏi: query gốc của benchmark; filter `region` chỉ áp dụng khi câu hỏi có `filter` do người dùng
  cung cấp (8/40 câu có đáp án: 6 dev, 1 test mới n06 ... và không câu thiếu evidence nào có filter).
- Hit: chunk chứa nguyên văn evidence gold, đúng nguồn + trang. Cutoff top-1 và top-3 cho mọi phương pháp.
- Context Evidence Rate (CER): evidence gold (đã chuẩn hoá ngắt dòng OCR) nằm trong context cuối gửi model.
- Thời gian: ms/truy vấn (median), máy RTX 3050 Ti 4 GB; chi phí: 0 USD (model local), ghi thêm token.

## Giai đoạn A — retrieval-only (dev)

| ID | Phương pháp | Giả thuyết |
|---|---|---|
| R1 | dense (pgvector E5) — baseline = cấu hình C Day 6 | mốc so sánh |
| R2 | bm25 | tốt ở câu có số liệu/tên riêng, kém ở câu paraphrase |
| R3 | hybrid RRF (dense+bm25) | bù nhau -> Hit@3 ≥ max(R1, R2) |
| R4 | dense + rerank cross-encoder | sửa hạng 1 (Hit@1) của dense |
| R5 | hybrid + rerank | kết hợp recall của hybrid + precision của rerank |

Tiêu chí chọn retriever cho agent (thứ tự ưu tiên, trên dev): Evidence Hit@3 → Hit@1 → MRR@10 → thời gian.

## Giai đoạn B — agent (dev)

| ID | Cấu hình | Giả thuyết |
|---|---|---|
| A0 | Agent baseline: tool dense, top-k 3, context "rank" 3000 ký tự, tối đa 3 lượt tìm | mốc |
| A1 | Cải thiện 1: tool dùng retriever chọn ở giai đoạn A | evidence vào context nhiều hơn -> trả lời đúng hơn |
| A2 | Cải thiện 2: thêm chiến lược context/abstention (quyết định sau khi đọc lỗi dev của A1) | sửa lỗi quan sát được |

Tiêu chí chọn cấu hình agent (thứ tự, trên dev):
1. Grounded answer trung bình /2 (câu có đáp án) — cao hơn tốt hơn;
2. Abstention đúng trên câu thiếu evidence, rồi số câu có đáp án bị từ chối nhầm (ít hơn);
3. Citation hợp lệ; 4. Thời gian/câu (median) thấp hơn.
Chênh lệch ≤ 1 câu ở tiêu chí 1 được coi là hoà -> xét tiêu chí sau.

## Bổ sung ~09:45 (trước mọi lần chạy agent trên benchmark dev)

- Smoke test 3 câu dev (d01, d13, u02) với model local Ollama (`qwen2.5:7b`, `qwen3:4b`) quá chậm trên GPU 4 GB
  (model 7B chỉ ~42% layer nằm trên GPU, ~40 s/câu; `qwen3:4b` lộ phần suy nghĩ vào câu trả lời).
  → Đổi LLM agent sang **Gemini API**. E5 + pgvector Day 6 và kết quả retrieval giữ nguyên.
- So sánh thêm **2 model agent** trên dev: `gemini-3.5-flash` vs `gemini-3.5-flash-lite`
  (cùng prompt, temperature 0, seed 7, `thinking_level="low"` — flash-lite không nhận `thinking_budget=0`).
- Thứ tự chạy dev: A0 (dense, flash) → A0-lite (dense, flash-lite) → chọn model theo tiêu chí agent ở trên
  → A1 (hybrid) → A2 (sau khi đọc lỗi A1). Smoke test không lưu và không tính vào kết quả.

## Test

Chỉ chạy sau khi ghi `results/selection.json`. Chạy baseline A0 và cấu hình chọn trên cùng test; không chỉnh gì sau đó.

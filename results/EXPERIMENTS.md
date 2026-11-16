# Nhật ký thí nghiệm Day 7 (theo thứ tự thời gian)

## 09:13 — Khoá benchmark
`data/benchmark/LOCK.json`. 46 câu: dev 30 có đáp án + 3 thiếu evidence; test 10 có đáp án (mới) + 3 thiếu evidence.

## 09:20 — Ghi kế hoạch & tiêu chí chọn: `results/PLAN.md`

## ~09:25 — Giai đoạn A: retrieval-only trên dev (query gốc, filter người dùng)

Lệnh: `python src/eval_retrieval.py --split dev` → `results/retrieval/dev/*.json`

| ID | Phương pháp | Hit@1 | Evidence Hit@3 | MRR@10 | Hit@10 | CER (top-3, 3000 ký tự) | Median ms/truy vấn |
|---|---|---|---|---|---|---|---|
| R1 | dense (pgvector E5) | 23/30 | 27/30 | 0.846 | 30/30 | 27/30 | 57 |
| R2 | bm25 | 19/30 | 24/30 | 0.739 | 28/30 | 24/30 | 1 |
| R3 | hybrid RRF | **25/30** | **28/30** | 0.895 | 30/30 | **28/30** | 59 |
| R4 | dense + rerank | 26/30 | 27/30 | 0.907 | 30/30 | 27/30 | 207 |
| R5 | hybrid + rerank | 26/30 | 27/30 | 0.907 | 30/30 | 27/30 | 206 |

- R1 tái lập đúng cấu hình C Day 6 (23/30, 27/30) → kết nối Day 6 đúng.
- Miss@3: dense {d06, d21, d26}; bm25 {d06, d08, d12, d19, d21, d22}; hybrid {d06, d19}; rerank {d06, d12, d20}.
- **Chọn theo tiêu chí đã ghi (Hit@3 → Hit@1 → MRR): R3 hybrid.** Rerank tăng Hit@1 (+1) nhưng giảm Hit@3
  (đẩy d12, d20 ra khỏi top-3) và chậm ×3.5 → không chọn ở giai đoạn này.

## ~09:35–09:53 — Giai đoạn B: agent trên dev (33 câu: 30 có đáp án + 3 thiếu evidence)

Nhãn chấm tay: `results/agent/dev/<cfg>/labels.json` (ngoại lệ có lý do) → `reviews.json`; bảng đọc: `review_sheet.md`.

| Cấu hình | Retriever / model / thay đổi | Grounded 0/1/2 (TB) | Citation hợp lệ | Abstain đúng | Từ chối nhầm | CER | Lượt tìm TB | Median s/câu | Token in+out |
|---|---|---|---|---|---|---|---|---|---|
| A0 | dense / 3.5-flash (baseline) | 4/1/25 (1.70) | 26/27 | 3/3 | 4/30 | 24/30 | 1.76 | 8.9 | 85,178+9,103 |
| A0-lite | dense / 3.5-flash-lite (so sánh model) | 4/1/25 (1.70) | 26/26 | 3/3 | 4/30 | 26/30 | 1.52 | 4.1 | 69,213+2,142 |
| A1 | **hybrid** / flash-lite (cải thiện 1) | 2/0/28 (1.87) | 29/29 | 3/3 | 2/30 | 29/30 | 1.30 | 2.1 | 57,534+1,930 |
| A2 | hybrid / flash-lite + **header không có species** (cải thiện 2) | 1/0/29 (1.93) | 32/32 | 3/3 | 0/30 | 29/30 | 1.21 | 2.2 | 54,322+1,882 |

Lỗi dev đã đọc:
1. **Budget đầy từ lượt tìm đầu (A0: d03, d17, d25, d26).** Query paraphrase lượt 1 kéo chunk lạc đề chiếm ~3000 ký tự;
   agent viết lại query đúng (tên loài) và retriever trả đúng trang gold, nhưng chunk bị `skipped` vì hết budget → từ chối.
   Giả thuyết A1: hybrid đưa evidence vào ngay lượt 1 → A1 sửa cả 4 câu (CER 24→29/30).
2. **Header `species` sai (A1: d10, d24).** Evidence ĐÃ trong context nhưng header ghi "section: Marbled cat" (thực ra
   đoạn cuối bài fishing cat) và "section: Puma" (thực ra Geoffroy's cat) → model lite tin header, từ chối.
   Lỗi metadata này đã ghi nhận ở Day 6. Giả thuyết A2: bỏ species khỏi header → A2 sửa cả 2 câu.
3. **Regression A2: d06** trả lời "High Atlas tới 1940s" (sư tử Bắc Phi) thay vì Gir Forest — mất tín hiệu tên bài.
4. A0-lite d10/d24: cùng lỗi header (model lite); A0 (flash) không mắc lỗi này nhưng mắc lỗi budget.

Chọn (09:53:36, `results/selection.json`): **A2**. A2 vs A1 chênh 1 câu → hoà tiêu chí 1; tiêu chí 2 từ chối nhầm 0/30 < 2/30.

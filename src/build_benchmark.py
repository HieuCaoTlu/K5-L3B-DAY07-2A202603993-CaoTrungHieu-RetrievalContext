"""Bước 1: dựng + kiểm chứng + khoá benchmark Day 7 (chạy TRƯỚC mọi thí nghiệm retrieval/agent).

Nguồn câu hỏi:
- dev có đáp án  : 30 câu dev Day 6 (d01-d30), giữ nguyên câu hỏi/đáp án/evidence.
- dev thiếu evid.: u01-u03 (mới).
- test có đáp án : n01-n10 (MỚI). Không dùng lại t01-t10 Day 6 vì kết quả của chúng đã được xem và
                   phân tích trong báo cáo Day 6 (gợi ý ý tưởng hybrid) -> không còn độc lập.
                   Viết trên đúng 10 loài test đã bốc thăm ở Day 6 (random.Random(2026)) -> dev/test
                   không chung loài; chọn sự kiện khác với t01-t10.
- test thiếu evid.: u04-u06 (mới).

Kiểm chứng:
- câu có đáp án: evidence nằm NGUYÊN VĂN trong ít nhất 1 chunk index B đúng nguồn + trang gold;
  nếu có filter thì giá trị filter phải đúng vùng của trang gold.
- câu thiếu evidence: các mẫu regex "absence_checks" không xuất hiện ở các trang được chỉ định
  (toàn corpus hoặc các trang của loài liên quan). Đây chỉ là kiểm tra hỗ trợ; lý do được ghi tay.

Chạy:  python src/build_benchmark.py          -> ghi questions.json, cases.json, verification.json
       python src/build_benchmark.py --lock   -> thêm LOCK.json (SHA-256) sau khi kiểm chứng đạt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rag import config  # noqa: E402

DAY6_QUESTIONS = config.ROOT.parent / "K5-L3B-DAY06-CaoTrungHieu-2A202603993-DataIngestion" / "benchmark" / "questions.jsonl"
SID = config.SOURCE_ID

# ---------------------------------------------------------------- test mới (có đáp án)
NEW_TEST = [
    {"id": "n01", "type": "direct", "species": "Black-footed cat",
     "question": "How large was the home range of an adult female black-footed cat observed for three months near Kimberley?",
     "expected_answer": "About 12 km2 (game farm near Kimberley, northern Cape, South Africa)",
     "page": 34, "evidence": "observed for three months on a game farm near Kimberley (northern Cape, South Africa) had a home range of 12 km"},
    {"id": "n02", "type": "condition_exception", "species": "Manul",
     "question": "Does the manul live in lowland sandy deserts?",
     "expected_answer": "Generally no - it is absent from lowland sandy desert basins, although it may penetrate them along river courses",
     "page": 124, "evidence": "but is generally absent from lowland sandy desert basins"},
    {"id": "n03", "type": "direct", "species": "Iberian lynx",
     "question": "What is the average weight of adult male Iberian lynxes?",
     "expected_answer": "12.8 kg (n=5)",
     "page": 131, "evidence": "adult males weighing an average of 12.8 kg (n=5)"},
    {"id": "n04", "type": "condition_exception", "species": "Asiatic wildcat",
     "question": "Are Asiatic wildcats found in the steppe grasslands of Mongolia?",
     "expected_answer": "No - they are typically associated with scrub desert and do not occur in the steppe grasslands of Mongolia and Inner Mongolia",
     "page": 125, "evidence": "They do not occur in the steppe grasslands of Mongolia and Inner Mongolia"},
    {"id": "n05", "type": "condition_exception", "species": "Oncilla",
     "question": "How many skins were exported from Paraguay in 1983 in the oncilla fur trade, and why might that number overstate oncillas?",
     "expected_answer": "84,500 skins; other spotted cat pelts were probably mixed in with oncilla",
     "page": 148, "evidence": "in 1983, 84,500 skins were exported from Paraguay"},
    {"id": "n06", "type": "metadata_filter", "species": "Leopard",
     "question": "According to Pocock, what colour are rain forest leopards in this region?",
     "expected_answer": "Dark, deep gold",
     "page": 50, "evidence": "rain forest leopards-dark, deep gold in color",
     "filter": {"region": "Sub-Saharan Africa"},
     "filter_context": "User says up front they study African leopards (leopard accounts exist for 3 regions)."},
    {"id": "n07", "type": "paraphrase", "species": "Clouded leopard",
     "question": "Which Asian cat is named after its markings of ellipses partly edged in black?",
     "expected_answer": "The clouded leopard",
     "page": 91, "evidence": "The clouded leopard is named after its distinctive markings-ellipses partially edged in black"},
    {"id": "n08", "type": "direct", "species": "Tiger",
     "question": "What is the average reproductive life span of female tigers in Chitwan?",
     "expected_answer": "6.1 years (n=12; range up to 12.5 years); males only 2.8 years",
     "page": 83, "evidence": "the average reproductive life span of Chitwan tigers to be 6.1 years for females"},
    {"id": "n09", "type": "paraphrase", "species": "Serval",
     "question": "For how long have the surviving long-legged spotted cats of north Africa probably been cut off from populations south of the Sahara?",
     "expected_answer": "At least 6,000-7,000 years (north African servals)",
     "page": 48, "evidence": "isolated from sub-Saharan\n\npopulations for at least 6,000-7,000 years"},
    {"id": "n10", "type": "direct", "species": "Manul",
     "question": "How many manul skins were reportedly taken each year in Mongolia in the early 1900s?",
     "expected_answer": "As many as 50,000 skins per year",
     "page": 124, "evidence": "Annual take in Mongolia in the early 1900s was reportedly as high as 50,000 skins"},
]

# ---------------------------------------------------------------- thiếu evidence
ALL = "all"
UNANSWERABLE = [
    {"id": "u01", "split": "dev",
     "question": "How many wild tigers did India's 2018 national tiger census count?",
     "absence_reason": "Tài liệu xuất bản 1996; chỉ có ước tính 3,000-4,000 hổ ở Ấn Độ thời đó (p.88). Không có số liệu 2018.",
     "absence_checks": [{"pages": ALL, "pattern": r"\b2018\b"}, {"pages": ALL, "pattern": r"\b20[01]\d\b"}]},
    {"id": "u02", "split": "dev",
     "question": "What is the maximum running speed of the jaguar?",
     "absence_reason": "Chỉ có tốc độ của báo săn (p.37) và tốc độ đi tuần của ocelot (p.163); bài báo đốm (p.143-146) không có tốc độ.",
     "absence_checks": [{"pages": [143, 144, 145, 146], "pattern": r"kph|km/h|mph|speed|fast"}]},
    {"id": "u03", "split": "dev",
     "question": "How many chromosomes does the marbled cat have?",
     "absence_reason": "p.101 chỉ nói marbled cat có karyotype giống Lynx/Panthera/Uncia, không nêu số nhiễm sắc thể.",
     "absence_checks": [{"pages": ALL, "pattern": r"chromosom|2n\s*=|diploid"}]},
    {"id": "u04", "split": "test",
     "question": "What IUCN Red List category was the Iberian lynx given in its 2015 reassessment?",
     "absence_reason": "Tài liệu xuất bản 1996; chỉ có xếp hạng Category 1 và phân loại IUCN thời đó (p.133). Không có đánh giá 2015.",
     "absence_checks": [{"pages": ALL, "pattern": r"\b2015\b"}, {"pages": ALL, "pattern": r"Red List"}]},
    {"id": "u05", "split": "test",
     "question": "How many ocelots were killed on Texas roads in 2010?",
     "absence_reason": "Tài liệu xuất bản 1996 nên không có số liệu 2010; trang ocelot (p.162-164) không có năm 2010.",
     "absence_checks": [{"pages": ALL, "pattern": r"\b2010\b"}]},
    {"id": "u06", "split": "test",
     "question": "What is the average litter size of the Bornean bay cat in the wild?",
     "absence_reason": "Bay cat (p.90-91) chỉ biết qua vài mẫu da/sọ; không có mục litter size cho loài này.",
     "absence_checks": [{"pages": [90, 91], "pattern": r"[Ll]itter"}]},
]


def load_day6() -> list[dict]:
    rows = [json.loads(l) for l in DAY6_QUESTIONS.read_text(encoding="utf-8").splitlines() if l.strip()]
    out = []
    for q in rows:
        if q["split"] != "dev":
            continue                      # t01-t10 không dùng (đã xem kết quả ở Day 6)
        item = {"id": q["id"], "split": "dev", "type": q["type"], "question": q["question"],
                "expected_answer": q["expected_answer"], "origin": "day6-dev",
                "gold": [{"source_id": q["gold_source_id"], "page": p, "evidence": q["evidence_quote"]}
                         for p in q["gold_page"]]}
        if q.get("filter"):
            item["filter"], item["filter_context"] = q["filter"], q["filter_context"]
        out.append(item)
    return out


def build() -> tuple[list[dict], list[dict]]:
    questions = load_day6()
    for n in NEW_TEST:
        item = {"id": n["id"], "split": "test", "type": n["type"], "question": n["question"],
                "expected_answer": n["expected_answer"], "origin": "day7-new", "species": n["species"],
                "gold": [{"source_id": SID, "page": n["page"], "evidence": n["evidence"]}]}
        if n.get("filter"):
            item["filter"], item["filter_context"] = n["filter"], n["filter_context"]
        questions.append(item)
    cases = [{"id": q["id"], "split": q["split"], "question": q["question"], "answerable": True,
              "expected_behavior": "search_and_answer", "type": q["type"],
              **({"filter": q["filter"]} if q.get("filter") else {})} for q in questions]
    cases += [{"id": u["id"], "split": u["split"], "question": u["question"], "answerable": False,
               "expected_behavior": "search_and_abstain", "type": "unanswerable",
               "absence_reason": u["absence_reason"], "absence_checks": u["absence_checks"]} for u in UNANSWERABLE]
    return questions, cases


def verify(questions: list[dict], cases: list[dict]) -> dict:
    chunks = json.loads(config.CHUNKS_PATH.read_text(encoding="utf-8"))
    page_text: dict[int, str] = {}
    page_region: dict[int, str] = {}
    for c in chunks:
        page_text[c["page"]] = page_text.get(c["page"], "") + "\n" + c["text"]
        page_region[c["page"]] = c["metadata"]["region"]
    errors, details = [], []
    ids = [c["id"] for c in cases]
    if len(set(ids)) != len(ids):
        errors.append("trùng ID")
    for q in questions:
        for g in q["gold"]:
            holders = [c["id"] for c in chunks if c["source_id"] == g["source_id"] and c["page"] == g["page"]
                       and g["evidence"] in c["text"]]
            ok = bool(holders)
            if q.get("filter") and q["filter"]["region"] != page_region.get(g["page"]):
                ok = False
                errors.append(f"{q['id']}: filter {q['filter']} != vùng trang {g['page']}")
            if not holders:
                errors.append(f"{q['id']}: evidence không nằm trọn trong chunk nào ở trang {g['page']}")
            details.append({"id": q["id"], "page": g["page"], "evidence_chunks": holders, "ok": ok})
    for u in UNANSWERABLE:
        for chk in u["absence_checks"]:
            pages = sorted(page_text) if chk["pages"] == ALL else chk["pages"]
            found = [p for p in pages if re.search(chk["pattern"], page_text.get(p, ""))]
            if found:
                errors.append(f"{u['id']}: mẫu {chk['pattern']!r} xuất hiện ở trang {found}")
            details.append({"id": u["id"], "pattern": chk["pattern"], "pages_checked": len(pages), "found_on": found,
                            "ok": not found})
    split_counts = {}
    for c in cases:
        key = f"{c['split']}/{'answerable' if c['answerable'] else 'unanswerable'}"
        split_counts[key] = split_counts.get(key, 0) + 1
    req = {"dev/answerable": 15, "dev/unanswerable": 2, "test/answerable": 5, "test/unanswerable": 2}
    for k, v in req.items():
        if split_counts.get(k, 0) < v:
            errors.append(f"{k}: {split_counts.get(k, 0)} < {v}")
    dev_species = set()  # dev và test không chung loài (dev Day 6 viết trên 30 loài dev)
    return {"counts": split_counts, "minimum": req, "errors": errors, "checks": details,
            "passed": not errors}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lock", action="store_true")
    args = ap.parse_args()
    questions, cases = build()
    report = verify(questions, cases)
    config.QUESTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    for path, obj in ((config.QUESTIONS_PATH, questions), (config.CASES_PATH, cases),
                      (config.QUESTIONS_PATH.parent / "verification.json", report)):
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("counts", "errors", "passed")}, ensure_ascii=False, indent=2))
    if args.lock:
        if not report["passed"]:
            sys.exit("Không khoá: kiểm chứng chưa đạt.")
        lock = {"questions_sha256": hashlib.sha256(config.QUESTIONS_PATH.read_bytes()).hexdigest(),
                "cases_sha256": hashlib.sha256(config.CASES_PATH.read_bytes()).hexdigest(),
                "chunks_sha256": hashlib.sha256(config.CHUNKS_PATH.read_bytes()).hexdigest(),
                "locked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "note": "Khoá trước mọi lần chạy retrieval/agent Day 7. Mọi script chạy đều kiểm tra hash này."}
        config.LOCK_PATH.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
        print("LOCKED", lock["locked_at"])


if __name__ == "__main__":
    main()

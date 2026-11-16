"""Tắt log/warning ồn của HF/transformers để output từng câu hiện ra ngay (không cần lọc bằng grep)."""
import logging
import os
import sys
import warnings

os.environ.setdefault("HF_HUB_OFFLINE", "1")          # model đã có trong cache, không gọi mạng
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")
for name in ("sentence_transformers", "transformers", "huggingface_hub", "httpx", "google_genai"):
    logging.getLogger(name).setLevel(logging.ERROR)
try:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

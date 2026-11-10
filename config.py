"""
config.py
---------
Cấu hình chung cho toàn bộ pipeline LLM-as-a-Judge.
"""

import os

# ── API ────────────────────────────────────────────────────────────────────────
GEMINI_API_KEY = "AIzaSyC_va_mfJfQNob6iQouwXN6vGpAIxZ8Vwg"

# Model Gemini dùng để judge (5 RPM / 20 RPD)
JUDGE_MODEL = "gemini-2.5-flash"

# Giới hạn rate: 5 requests/phút → sleep 12 giây mỗi request để an toàn
RATE_LIMIT_SLEEP = 13      # giây giữa mỗi lần gọi API

# ── Dataset HuggingFace ────────────────────────────────────────────────────────
HF_LLM_JUDGE_DATASET   = "Lakshan2003/gemini-2.5-flash-customerservice-LLM-as-a-judge-data"
HF_HUMAN_EVAL_DATASETS = [
    "Lakshan2003/Gemini-2.5-Flash-customerservice-Human-evaluator_1_data",
    "Lakshan2003/Gemini-2.5-Flash-customerservice-Human-evaluator_2_data",
    "Lakshan2003/Gemini-2.5-Flash-customerservice-Human-evaluator_3_data",
]

# ── Cột đánh giá ──────────────────────────────────────────────────────────────
SCORE_COLS = [
    "Human-Likeness",
    "Continuity and Context Understanding",
    "Tone and Clarity",
    "Task Appropriateness",
]

# ── Thư mục lưu dữ liệu ───────────────────────────────────────────────────────
DATA_DIR    = "data"
RESULTS_DIR = "results"

# Tên các file output
MERGED_DATASET_PATH       = os.path.join(DATA_DIR, "merged_dataset.csv")
BASELINE_RESULTS_PATH     = os.path.join(RESULTS_DIR, "baseline_results.csv")
IMPROVED_RESULTS_PATH     = os.path.join(RESULTS_DIR, "improved_results.csv")
METRICS_COMPARISON_PATH   = os.path.join(RESULTS_DIR, "metrics_comparison.csv")

# ── Số mẫu tối đa gọi API (vì giới hạn 20 RPD) ───────────────────────────────
# Mỗi pipeline (baseline + improved) gọi N mẫu → tổng ≤ 20
MAX_SAMPLES_PER_RUN = 9    # 9 baseline + 9 improved = 18 calls/ngày (an toàn)

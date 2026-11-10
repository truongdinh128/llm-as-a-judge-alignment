"""
metrics_utils.py
----------------
Các hàm tính toán metrics để đánh giá mức độ tương đồng giữa
điểm số LLM-as-a-Judge và điểm Human Evaluation.

Metrics được sử dụng:
  1. Pearson Correlation  – đo quan hệ tuyến tính
  2. Spearman Correlation – đo quan hệ đơn điệu (rank-based)
  3. Kendall's Tau        – đo nhất quán về thứ tự
  4. MAE (Mean Absolute Error)
  5. RMSE (Root Mean Squared Error)
  6. Exact Match Rate     – tỷ lệ điểm hoàn toàn khớp (làm tròn 0.5)
  7. Within-1 Accuracy    – tỷ lệ điểm sai lệch ≤ 1
  8. Cohen's Kappa (weighted) – thỏa thuận giữa judge và human (rounded)
  9. Bias                 – chênh lệch trung bình (judge - human)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score


def _slugify(col: str) -> str:
    """'Continuity and Context Understanding' → 'Continuity_and_Context_Understanding'."""
    import re
    return re.sub(r"[^a-zA-Z0-9]+", "_", col).strip("_")


# Public alias
slugify_col = _slugify


def _safe_round(arr: np.ndarray) -> np.ndarray:
    """Làm tròn điểm thực về số nguyên gần nhất (clip về [1, 5])."""
    return np.clip(np.round(arr).astype(int), 1, 5)


def pearson_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Hệ số tương quan Pearson."""
    if len(y_true) < 2:
        return float("nan")
    r, _ = stats.pearsonr(y_true, y_pred)
    return float(r)


def spearman_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Hệ số tương quan Spearman (rank-based)."""
    if len(y_true) < 2:
        return float("nan")
    r, _ = stats.spearmanr(y_true, y_pred)
    return float(r)


def kendall_tau(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Kendall's Tau-b."""
    if len(y_true) < 2:
        return float("nan")
    tau, _ = stats.kendalltau(y_true, y_pred)
    return float(tau)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def exact_match_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Tỷ lệ điểm làm tròn trùng hoàn toàn với human."""
    return float(np.mean(_safe_round(y_true) == _safe_round(y_pred)))


def within_one_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Tỷ lệ |pred_round - true_round| ≤ 1."""
    return float(np.mean(np.abs(_safe_round(y_true) - _safe_round(y_pred)) <= 1))


def weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Cohen's Kappa với trọng số quadratic (cần giá trị integer 1-5)."""
    t = _safe_round(y_true)
    p = _safe_round(y_pred)
    try:
        return float(cohen_kappa_score(t, p, weights="quadratic",
                                       labels=list(range(1, 6))))
    except Exception:
        return float("nan")


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Độ lệch trung bình (judge - human). Dương → judge cao hơn human."""
    return float(np.mean(y_pred - y_true))


# ── Tổng hợp tất cả metrics ───────────────────────────────────────────────────

def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str = "",
) -> dict[str, float]:
    """
    Tính toàn bộ metrics và trả về dict.

    Parameters
    ----------
    y_true : điểm human (ground truth)
    y_pred : điểm judge (prediction)
    prefix : tiền tố key (ví dụ tên metric / tên cột)
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    # Lọc bỏ NaN
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]

    if len(y_true) == 0:
        return {}

    p = f"{prefix}_" if prefix else ""
    return {
        f"{p}pearson":        pearson_corr(y_true, y_pred),
        f"{p}spearman":       spearman_corr(y_true, y_pred),
        f"{p}kendall_tau":    kendall_tau(y_true, y_pred),
        f"{p}mae":            mae(y_true, y_pred),
        f"{p}rmse":           rmse(y_true, y_pred),
        f"{p}exact_match":    exact_match_rate(y_true, y_pred),
        f"{p}within_1_acc":   within_one_accuracy(y_true, y_pred),
        f"{p}weighted_kappa": weighted_kappa(y_true, y_pred),
        f"{p}bias":           bias(y_true, y_pred),
    }


def compute_metrics_per_criterion(
    df: pd.DataFrame,
    human_cols: list[str],
    judge_cols: list[str],
) -> pd.DataFrame:
    """
    Tính metrics cho từng tiêu chí đánh giá.

    Parameters
    ----------
    df         : DataFrame chứa cả cột human và judge
    human_cols : danh sách cột điểm human (theo thứ tự tiêu chí)
    judge_cols : danh sách cột điểm judge (theo thứ tự tiêu chí)

    Returns
    -------
    DataFrame với index = tên tiêu chí, columns = tên metric
    """
    records = []
    for hcol, jcol in zip(human_cols, judge_cols):
        sub = df[[hcol, jcol]].dropna()
        m = compute_all_metrics(sub[hcol].values, sub[jcol].values)
        m["criterion"] = hcol.replace("human_", "").replace("_", " ")
        records.append(m)

    result = pd.DataFrame(records).set_index("criterion")
    return result


def summarize_metrics(
    df: pd.DataFrame,
    human_cols: list[str],
    judge_cols: list[str],
) -> dict[str, float]:
    """Tính macro-average metrics trên tất cả tiêu chí."""
    all_true = np.concatenate([df[c].dropna().values for c in human_cols])
    all_pred = np.concatenate([
        df[jc].loc[df[hc].notna()].values
        for hc, jc in zip(human_cols, judge_cols)
    ])
    return compute_all_metrics(all_true, all_pred, prefix="overall")

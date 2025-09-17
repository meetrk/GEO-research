from typing import Sequence, Tuple, List
import pandas as pd
import numpy as np
from typing import Union, Sequence, List
from tqdm.auto import tqdm
import math

Score = Tuple[float, float]
ScoreList = Sequence[Score]

def evaluate_diff(old_scores: ScoreList, new_scores: ScoreList, *, strict: bool = True) -> List[Score]:
    """
    Pairwise percentage change between new and old scores.
    Each score is a tuple: (importance, positive_word_count).
    Returns a list of (importance_pct_change, positive_word_count_pct_change), rounded to 2 decimals.
    Percentage change = ((new - old) / old) * 100
    If old == 0:
        - returns 0 if new == 0
        - returns inf if new != 0
    If strict is True, raises ValueError on length mismatch; otherwise trims to min length.
    """
    if strict and len(old_scores) != len(new_scores):
        raise ValueError(f"Length mismatch: old={len(old_scores)} new={len(new_scores)}")
    n = min(len(old_scores), len(new_scores))
    
    def pct_change(old: float, new: float) -> float:
        if old == 0:
            return 0.0 if new == 0 else float('inf')
        return round(((new - old) / old) * 100.0, 2)

    return [(pct_change(old_scores[i][0], new_scores[i][0]),
             pct_change(old_scores[i][1], new_scores[i][1])) for i in range(n)]

def batch_evaluate_diff(
    df_or_dfs: Union[pd.DataFrame, Sequence[pd.DataFrame]],
    old_results_col: str = 'evaluation_results',
    new_results_col: str = 'evaluation_results_new',
    output_col: str = 'evaluation_diff',
    *,
    strict: bool = True,
    show_progress: bool = False,
    concat: bool = True,
    batch_id_col: str = 'batch_id',
    reset_index: bool = True,
) -> Union[pd.DataFrame, List[pd.DataFrame]]:
    """
    Compute per-row differences and store them in `output_col`.

    - If df_or_dfs is a DataFrame, returns a processed DataFrame.
    - If df_or_dfs is a sequence of DataFrames (batches):
        - If concat=True, returns a single concatenated DataFrame with a `batch_id` column.
        - If concat=False, returns a list of processed DataFrames (one per batch).
    """

    def _process_one(dfx: pd.DataFrame) -> pd.DataFrame:
        required = {old_results_col, new_results_col}
        missing = [c for c in required if c not in dfx.columns]
        if missing:
            raise KeyError(f"Missing columns: {missing}")
        out = dfx.copy()
        if show_progress:
            tqdm.pandas(desc="Computing evaluation differences")
            apply = out.progress_apply
        else:
            apply = out.apply

        def _row_diff(row):
            return evaluate_diff(row[old_results_col], row[new_results_col], strict=strict)

        out[output_col] = apply(_row_diff, axis=1)  # type: ignore
        return out

    # Single DataFrame case
    if isinstance(df_or_dfs, pd.DataFrame):
        return _process_one(df_or_dfs)

    # Multiple batches case
    dfs = list(df_or_dfs)
    processed: List[pd.DataFrame] = []

    iterator = range(len(dfs))
    if show_progress:
        iterator = tqdm(iterator, desc="Processing batches")

    for i in iterator:
        out_i = _process_one(dfs[i])
        if concat:
            out_i = out_i.copy()
            out_i[batch_id_col] = i
        processed.append(out_i)

    if concat:
        return pd.concat(processed, ignore_index=reset_index) if reset_index else pd.concat(processed)
    return processed

def summarize_differences(
    df: pd.DataFrame,
    diff_col: str = 'evaluation_diff',
    index_col: str = 'choosen_doc_idx',
    *,
    weight_imp: float = 0.5,
    weight_wc: float = 0.5
) -> pd.Series:
    """
    Aggregate per-row chosen document deltas into summary metrics.
    Returns a pandas Series with means, medians, stds, and positive rates.
    All values rounded to 2 decimals.
    """
    if abs(weight_imp + weight_wc - 1.0) > 1e-9:
        total = weight_imp + weight_wc
        weight_imp, weight_wc = weight_imp / total, weight_wc / total

    imp_deltas: List[float] = []
    wc_deltas: List[float] = []
    totals: List[float] = []

    for _, row in df.iterrows():
        diffs = row.get(diff_col)
        idx = row.get(index_col)
        if diffs is None or not isinstance(idx, int) or not (0 <= idx < len(diffs)):
            continue
        imp_delta, wc_delta = diffs[idx]
        imp_deltas.append(float(imp_delta))
        wc_deltas.append(float(wc_delta))
        totals.append(weight_imp * float(imp_delta) + weight_wc * float(wc_delta))

    if not imp_deltas:
        raise ValueError("No valid differences found to summarize.")

    imp = np.array(imp_deltas, dtype=float)
    wc = np.array(wc_deltas, dtype=float)
    tot = np.array(totals, dtype=float)

    return pd.Series(
        {
            "n": len(imp),
            "mean_importance": round(float(np.mean(imp)), 2),
            "mean_word_count": round(float(np.mean(wc)), 2),
            "mean_total": round(float(np.mean(tot)), 2),
            "median_importance": round(float(np.median(imp)), 2),
            "median_word_count": round(float(np.median(wc)), 2),
            "std_importance": round(float(np.std(imp, ddof=1)), 2) if len(imp) > 1 else 0.0,
            "std_word_count": round(float(np.std(wc, ddof=1)), 2) if len(wc) > 1 else 0.0,
            "positive_rate_importance": round(float(np.mean(imp > 0)), 2),
            "positive_rate_word_count": round(float(np.mean(wc > 0)), 2),
            "positive_rate_total": round(float(np.mean(tot > 0)), 2),
            "weight_importance": round(weight_imp, 2),
            "weight_word_count": round(weight_wc, 2),
        }
    )

def print_diff_summary(summary, method_name="Method", width=72):
    """
    Pretty-print the summary returned by summarize_differences with a method name.
    All values rounded to 2 decimals.
    """
    # Build header
    title = f"[{method_name}] Evaluation Summary"
    width = max(width, len(title) + 4)
    sep = "=" * width

    n = int(summary.get("n", 0))

    # Prefer explicit counts if provided
    pos = summary.get("positive_count_total")
    neg = summary.get("negative_count_total")

    if pos is None:
        pr_total = float(summary.get("positive_rate_total", 0.0))
        pos = int(round(n * pr_total))
    else:
        pos = int(pos)

    if neg is None:
        neg = int(n - pos)
    else:
        neg = int(neg)

    # Extract values with safe defaults
    def g(k, default=0.0):
        return float(summary.get(k, default))

    n = int(summary.get("n", 0))
    mean_imp = g("mean_importance")
    mean_wc = g("mean_word_count")
    mean_total = g("mean_total")
    median_imp = g("median_importance")
    median_wc = g("median_word_count")
    std_imp = g("std_importance")
    std_wc = g("std_word_count")

    # Print
    print(sep)
    print(title.center(width))
    print(sep)
    label_w = 18
    print(f"{'n rows':<{label_w}} {n:>10d}")
    print(f"{'mean':<{label_w}} importance={mean_imp:.2f}, word_count={mean_wc:.2f}, total={mean_total:.2f}")
    print(f"{'median':<{label_w}} importance={median_imp:.2f}, word_count={median_wc:.2f}")
    print(f"{'std':<{label_w}} importance={std_imp:.2f}, word_count={std_wc:.2f}")
    print(f"{'overall sentiment':<{label_w}} overall positive:{pos}, overall negative:{neg}")
    print(sep)

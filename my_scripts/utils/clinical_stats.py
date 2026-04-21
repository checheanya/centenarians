"""Univariate clinical tests vs Condition; tables, FDR, printed summary."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


def _clinical_present(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def _filter_two_groups(
    df: pd.DataFrame,
    condition_col: str,
    allowed: tuple[str, ...] = ("Control", "Centenarian"),
) -> pd.DataFrame:
    s = df[condition_col].astype(str).str.strip()
    return df.loc[s.isin(allowed)].copy()


def _two_group_continuous_p(
    a: np.ndarray,
    b: np.ndarray,
    *,
    method: Literal["mannwhitney", "ttest"] = "mannwhitney",
) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    if method == "ttest":
        _, p = stats.ttest_ind(a, b, equal_var=False)
        return float(p)
    try:
        _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        return float(p)
    except ValueError:
        return np.nan


def _chi2_independence_p(sub: pd.DataFrame, row_col: str, col_col: str) -> float:
    ct = pd.crosstab(sub[row_col], sub[col_col])
    if ct.size < 4 or min(ct.shape) < 2:
        return np.nan
    try:
        _, p, _, _ = stats.chi2_contingency(ct)
        return float(p)
    except ValueError:
        return np.nan


def _classify_plot_kind(
    d: pd.DataFrame,
    col: str,
    condition_col: str,
) -> str:
    if col == condition_col:
        return "counts"
    if col == "Gender" or (d[col].dtype == object and d[col].nunique(dropna=True) <= 6):
        return "count_hue"
    return "box"


def univariate_tests_for_column_list(
    df: pd.DataFrame,
    columns: list[str],
    *,
    category: str,
    condition_col: str = "Condition",
    order: tuple[str, ...] = ("Control", "Centenarian"),
    continuous_test: Literal["mannwhitney", "ttest"] = "mannwhitney",
) -> pd.DataFrame:
    cols = _clinical_present(df, columns)
    d = _filter_two_groups(df, condition_col)
    rows: list[dict] = []

    for col in cols:
        kind = _classify_plot_kind(d, col, condition_col)
        if kind == "counts":
            rows.append(
                {
                    "category": category,
                    "variable": col,
                    "test": "— (descriptive N only)",
                    "p_value": np.nan,
                    "note": "bar chart of N per group",
                }
            )
            continue
        if kind == "count_hue":
            sub = d[[condition_col, col]].dropna()
            p = _chi2_independence_p(sub, col, condition_col)
            rows.append(
                {
                    "category": category,
                    "variable": col,
                    "test": "Chi-square independence",
                    "p_value": p,
                    "note": "",
                }
            )
            continue
        sub = d[[condition_col, col]].copy()
        sub[col] = pd.to_numeric(sub[col], errors="coerce")
        sub = sub.dropna(subset=[col])
        g0 = sub[sub[condition_col].astype(str).str.strip() == order[0]][col].values
        g1 = sub[sub[condition_col].astype(str).str.strip() == order[1]][col].values
        p = _two_group_continuous_p(g0, g1, method=continuous_test)
        tname = (
            "Mann-Whitney U"
            if continuous_test == "mannwhitney"
            else "Welch t-test"
        )
        rows.append(
            {
                "category": category,
                "variable": col,
                "test": tname,
                "p_value": p,
                "note": "",
            }
        )

    return pd.DataFrame(rows)


def add_fdr_within_category(results: pd.DataFrame) -> pd.DataFrame:
    out = results.copy()
    out["FDR"] = np.nan
    for cat in out["category"].dropna().unique():
        m = out["category"] == cat
        pv = out.loc[m, "p_value"].astype(float)
        valid = pv.notna() & np.isfinite(pv)
        if valid.sum() == 0:
            continue
        idx = out.index[m][valid]
        _, fdr, _, _ = multipletests(pv.loc[idx], method="fdr_bh")
        out.loc[idx, "FDR"] = fdr
    return out


def format_significant_summary(df: pd.DataFrame, *, fdr_alpha: float = 0.05) -> str:
    sub = df[
        df["FDR"].notna()
        & (df["FDR"] < fdr_alpha)
        & (df["test"] != "— (descriptive N only)")
    ].sort_values(["category", "FDR"])
    if sub.empty:
        return (
            f"No variables reached FDR < {fdr_alpha} within their category "
            "(Benjamini–Hochberg per category).\n"
        )
    lines = [
        f"Significant differences (FDR < {fdr_alpha}; BH correction within each category):",
        "",
    ]
    for cat in sub["category"].unique():
        lines.append(f"## {cat}")
        part = sub[sub["category"] == cat]
        for _, r in part.iterrows():
            lines.append(
                f"  - {r['variable']}: {r['test']}, p={r['p_value']:.4g}, FDR={r['FDR']:.4g}"
            )
        lines.append("")
    return "\n".join(lines)


def run_clinical_univariate_export(
    meta_full: pd.DataFrame,
    *,
    to_demographic: list[str],
    to_clinical: list[str],
    to_blood: list[str],
    condition_col: str = "Condition",
    continuous_test: Literal["mannwhitney", "ttest"] = "mannwhitney",
    out_csv: str | Path | None = None,
    out_summary: str | Path | None = None,
    fdr_alpha: float = 0.05,
    print_table: bool = True,
) -> tuple[pd.DataFrame, str]:
    parts = []
    if to_demographic:
        parts.append(
            univariate_tests_for_column_list(
                meta_full,
                to_demographic,
                category="demographic",
                condition_col=condition_col,
                continuous_test=continuous_test,
            )
        )
    if to_clinical:
        parts.append(
            univariate_tests_for_column_list(
                meta_full,
                to_clinical,
                category="clinical",
                condition_col=condition_col,
                continuous_test=continuous_test,
            )
        )
    if to_blood:
        parts.append(
            univariate_tests_for_column_list(
                meta_full,
                to_blood,
                category="blood",
                condition_col=condition_col,
                continuous_test=continuous_test,
            )
        )
    if not parts:
        empty = pd.DataFrame(
            columns=["category", "variable", "test", "p_value", "FDR", "note"]
        )
        return empty, "No columns to test.\n"

    combined = pd.concat(parts, ignore_index=True)
    combined = add_fdr_within_category(combined)
    summary = format_significant_summary(combined, fdr_alpha=fdr_alpha)

    if out_csv is not None:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(out_csv, index=False)

    if out_summary is not None:
        out_summary = Path(out_summary)
        out_summary.parent.mkdir(parents=True, exist_ok=True)
        header = (
            "Univariate association with Condition (Control vs Centenarian).\n"
            "FDR: Benjamini–Hochberg within each category (demographic / clinical / blood).\n\n"
        )
        out_summary.write_text(header + summary, encoding="utf-8")

    if print_table:
        pd.set_option("display.max_rows", 200)
        pd.set_option("display.width", 120)
        print("\n=== Clinical univariate tests (all variables) ===\n")
        print(combined.to_string(index=False))
        print("\n=== Summary (FDR < %.4f) ===\n" % fdr_alpha)
        print(summary)

    return combined, summary

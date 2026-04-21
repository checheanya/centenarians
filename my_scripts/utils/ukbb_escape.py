"""UKBB age–protein table: gene preprocessing for CIAO matching and Age Beta summary plots."""

from __future__ import annotations

import inspect
import tempfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import patheffects as mpe
from matplotlib.patches import Patch

# Category colours aligned with sanju_version/.../make_figure3_v3.py (Nature Aging style).
ESCAPE_CATEGORY_COLORS: dict[str, str] = {
    "NS": "#CCCCCC",
    "Aging only": "#F4C891",
    "Centenarian only": "#B07AA1",
    "Concordant": "#4DBBD5",
    "Escape": "#F39B7F",
    "Reversed": "#91D1C2",
}


def preprocess_ukbb_protein_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add columns for joining to centenarian / DIA DE outputs.

    - ``gene_symbol_match``: uppercase symbol from ``Gene`` (first token if ``;``-separated).
    - ``protein_id_stem``: field before first ``:`` in ``ProteinID`` (Olink assay id).
    - ``uniprot_id``: second ``:``-separated field when present (UniProt accession).
    """
    out = df.copy()
    g = out["Gene"].astype(str).str.strip()
    out["gene_symbol_match"] = g.str.split(";").str[0].str.strip().str.upper()

    pid = out["ProteinID"].astype(str)
    sp = pid.str.split(":", expand=True)
    out["protein_id_stem"] = sp[0].str.strip().str.upper()
    if sp.shape[1] > 1:
        out["uniprot_id"] = sp[1].fillna("").astype(str).str.strip()
    else:
        out["uniprot_id"] = ""

    return out


def plot_ukbb_age_beta_sign_bar(
    df: pd.DataFrame,
    out_path: str | Path | None = None,
    *,
    beta_col: str = "Age_Beta",
    title: str = "UKBB proteins (Age p ≤ 0.05): sign of Age Beta",
    show: bool = False,
) -> Path | None:
    """Bar chart: counts of positive vs negative Age Beta (same red/blue semantics as DE LFC plot)."""
    pos = int((df[beta_col] > 0).sum())
    neg = int((df[beta_col] < 0).sum())
    zero = int((df[beta_col] == 0).sum())

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    cats = ["Positive Age Beta\n(increase with age)", "Negative Age Beta\n(decrease with age)"]
    vals = [pos, neg]
    colors = ["#c0392b", "#2980b9"]
    x = np.arange(len(cats))
    ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=9)
    ax.set_ylabel("Number of proteins", fontsize=10)
    ax.set_title(title, fontsize=11)
    ymax = max(vals) if max(vals) > 0 else 1
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02 * ymax, str(v), ha="center", va="bottom", fontsize=10)
    if zero:
        ax.text(
            0.98,
            0.02,
            f"Age Beta = 0: {zero}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="0.4",
        )
    sns.despine(ax=ax)
    plt.tight_layout()
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    plt.close(fig)
    return Path(out_path) if out_path is not None else None


def plot_ukbb_top_age_beta_barh(
    df: pd.DataFrame,
    out_path: str | Path | None = None,
    *,
    beta_col: str = "Age_Beta",
    gene_col: str = "gene_symbol_match",
    top_n: int = 10,
    title: str = "UKBB: top Age Beta (p ≤ 0.05)",
    show: bool = False,
) -> Path | None:
    """
    Horizontal bars: top ``top_n`` positive and top ``top_n`` negative Age Beta (DE ``plot_top_lfc_barplot`` style).
    """
    sub = df.dropna(subset=[beta_col, gene_col]).copy()
    if sub.empty:
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.text(0.5, 0.5, "No rows to plot", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        plt.tight_layout()
        if out_path is not None:
            out_path = Path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
        if show:
            plt.show()
        plt.close(fig)
        return Path(out_path) if out_path is not None else None

    up = sub[sub[beta_col] > 0].sort_values(beta_col, ascending=False).head(top_n)
    down = sub[sub[beta_col] < 0].sort_values(beta_col, ascending=True).head(top_n)
    combined = pd.concat([up, down])
    combined = combined.sort_values(beta_col, ascending=False)

    names = combined[gene_col].astype(str).values
    vals = combined[beta_col].astype(float).values
    colors = np.where(vals > 0, "#c0392b", "#2980b9")
    y = np.arange(len(combined))

    fig_h = max(4.0, 0.38 * len(combined) + 1.6)
    fig, ax = plt.subplots(figsize=(8, fig_h))
    ax.barh(y, vals, color=colors, edgecolor="white", linewidth=0.4, height=0.72)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.axvline(0, color="0.35", lw=0.7)
    ax.set_xlabel("Age Beta (UKBB; per-year change on modeled scale)", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.invert_yaxis()
    span = max(float(np.nanmax(np.abs(vals))), 1e-9)
    pad = 0.06 * span
    for yi, v in zip(y, vals):
        x_text = v + pad if v >= 0 else v - pad
        ax.text(x_text, yi, f"{v:.4f}", va="center", ha="left" if v >= 0 else "right", fontsize=7)
    ax.legend(
        handles=[
            Patch(facecolor="#c0392b", label=f"Largest positive Age Beta (n≤{top_n})"),
            Patch(facecolor="#2980b9", label=f"Most negative Age Beta (n≤{top_n})"),
        ],
        frameon=False,
        loc="lower right",
        fontsize=8,
    )
    sns.despine(ax=ax)
    plt.tight_layout()
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    plt.close(fig)
    return Path(out_path) if out_path is not None else None


def _first_gene_symbol(x: object) -> str:
    if pd.isna(x):
        return ""
    return str(x).split(";")[0].strip().upper()


def de_gene_count_for_merge(
    de_res: pd.DataFrame,
    *,
    gene_col_de: str = "PG.Genes",
    padj_max: float | None = None,
) -> int:
    """Number of DE genes after the same prep as :func:`merge_ukbb_centenarian_de` (optional ``padj`` filter, then dedup)."""
    de = de_res.copy()
    if padj_max is not None:
        if "padj" not in de.columns:
            raise ValueError("de_res must contain 'padj' when padj_max is set")
        de = de[pd.to_numeric(de["padj"], errors="coerce") <= padj_max].copy()
    de["_sym"] = de[gene_col_de].map(_first_gene_symbol)
    de = de[de["_sym"].str.len() > 0]
    return int(de.drop_duplicates("_sym", keep="first").shape[0])


def merge_ukbb_centenarian_de(
    ukbb_preprocessed: pd.DataFrame,
    de_res: pd.DataFrame,
    *,
    age_p_max: float | None = 0.05,
    de_padj_max: float | None = None,
    gene_col_de: str = "PG.Genes",
) -> pd.DataFrame:
    """
    Inner join on gene symbol: UKBB (preprocessed) × centenarian DE.

    DE rows can be restricted with ``de_padj_max`` (e.g. ``0.05``); then duplicates collapse to one row per symbol.

    If ``age_p_max`` is ``None``, **no** UKBB Age p-value filter is applied (full UKBB table).
    If set (default ``0.05``), only UKBB rows with ``Age_pvalue <= age_p_max`` are kept before the join.

    Adds ``Gene`` (match key), ``Log2FC``, ``FDR`` (from ``padj``), ``Cent_sig``, ``Age_sig``.
    """
    de = de_res.copy()
    if de_padj_max is not None:
        if "padj" not in de.columns:
            raise ValueError("de_res must contain 'padj' when de_padj_max is set")
        de = de[pd.to_numeric(de["padj"], errors="coerce") <= de_padj_max].copy()
    de["_sym"] = de[gene_col_de].map(_first_gene_symbol)
    de = de[de["_sym"].str.len() > 0]
    de = de.drop_duplicates("_sym", keep="first")
    de = de.rename(columns={"log2FC": "Log2FC", "padj": "FDR"})
    if age_p_max is None:
        sub = ukbb_preprocessed.copy()
    else:
        sub = ukbb_preprocessed[ukbb_preprocessed["Age_pvalue"] <= age_p_max].copy()
    merged = sub.merge(
        de[["_sym", "Log2FC", "FDR", "PG.Genes"]],
        left_on="gene_symbol_match",
        right_on="_sym",
        how="inner",
    )
    merged["Gene"] = merged["gene_symbol_match"].astype(str)
    merged["Cent_sig"] = merged["FDR"].astype(float) < 0.05
    if "Age_significant" in merged.columns:
        merged["Age_sig"] = merged["Age_significant"].apply(lambda x: x is True or x == "True" or x == True)
    else:
        merged["Age_sig"] = True
    return merged.drop(columns=["_sym"], errors="ignore")


def escape_overlap_report(
    ukbb_preprocessed: pd.DataFrame,
    de_res: pd.DataFrame,
    *,
    age_p_max: float | None,
    de_padj_max: float | None = None,
    label: str = "overlap",
) -> dict[str, Any]:
    """
    Run the same inner merge as :func:`merge_ukbb_centenarian_de` and return only
    :func:`summarize_escape_merge` stats (the merged frame is not returned).

    Use this for stepwise reporting (e.g. unfiltered overlap) without keeping an intermediate ``merged`` variable.
    """
    merged = merge_ukbb_centenarian_de(
        ukbb_preprocessed,
        de_res,
        age_p_max=age_p_max,
        de_padj_max=de_padj_max,
    )
    if age_p_max is None:
        n_u = len(ukbb_preprocessed)
    else:
        n_u = int((ukbb_preprocessed["Age_pvalue"] <= age_p_max).sum())
    n_d = de_gene_count_for_merge(de_res, padj_max=de_padj_max)
    return summarize_escape_merge(
        merged,
        label=label,
        n_ukbb_rows_input=n_u,
        n_de_rows_input=n_d,
    )


def summarize_escape_merge(
    merged: pd.DataFrame,
    *,
    label: str = "merge",
    n_ukbb_rows_input: int | None = None,
    n_de_rows_input: int | None = None,
) -> dict[str, Any]:
    """
    Descriptive stats for an overlap table (no classification): sizes, Age Beta / log2FC summaries, correlations.

    Pass ``n_ukbb_rows_input`` / ``n_de_rows_input`` for coverage percentages (rows in each input before join).
    """
    from scipy.stats import pearsonr, spearmanr

    d: dict[str, Any] = {"label": label, "n_overlap": int(len(merged))}
    if n_ukbb_rows_input is not None and n_ukbb_rows_input > 0:
        d["n_ukbb_rows_input"] = int(n_ukbb_rows_input)
        d["pct_ukbb_rows_in_overlap"] = 100.0 * len(merged) / n_ukbb_rows_input
    if n_de_rows_input is not None and n_de_rows_input > 0:
        d["n_de_rows_input"] = int(n_de_rows_input)
        d["pct_de_rows_in_overlap"] = 100.0 * len(merged) / n_de_rows_input

    if merged.empty:
        return d

    ab = pd.to_numeric(merged["Age_Beta"], errors="coerce")
    lfc = pd.to_numeric(merged["Log2FC"], errors="coerce")
    d["age_beta_n_non_na"] = int(ab.notna().sum())
    d["log2fc_n_non_na"] = int(lfc.notna().sum())
    d["age_beta_mean"] = float(ab.mean())
    d["age_beta_median"] = float(ab.median())
    d["age_beta_sd"] = float(ab.std(ddof=1)) if ab.notna().sum() > 1 else float("nan")
    d["age_beta_min"] = float(ab.min())
    d["age_beta_max"] = float(ab.max())
    d["age_beta_n_pos"] = int((ab > 0).sum())
    d["age_beta_n_neg"] = int((ab < 0).sum())
    d["age_beta_n_zero"] = int((ab == 0).sum())

    d["log2fc_mean"] = float(lfc.mean())
    d["log2fc_median"] = float(lfc.median())
    d["log2fc_sd"] = float(lfc.std(ddof=1)) if lfc.notna().sum() > 1 else float("nan")
    d["log2fc_min"] = float(lfc.min())
    d["log2fc_max"] = float(lfc.max())
    d["log2fc_n_pos"] = int((lfc > 0).sum())
    d["log2fc_n_neg"] = int((lfc < 0).sum())
    d["log2fc_n_zero"] = int((lfc == 0).sum())

    if "FDR" in merged.columns:
        fd = pd.to_numeric(merged["FDR"], errors="coerce")
        d["n_fdr_lt_005"] = int((fd < 0.05).sum())
        d["n_fdr_lt_001"] = int((fd < 0.01).sum())
    if "Age_pvalue" in merged.columns:
        ap = pd.to_numeric(merged["Age_pvalue"], errors="coerce")
        d["n_age_p_lt_005"] = int((ap <= 0.05).sum())
        d["n_age_p_lt_001"] = int((ap <= 0.001).sum())
    if "Cent_sig" in merged.columns:
        d["n_cent_sig"] = int(merged["Cent_sig"].astype(bool).sum())
    if "Age_sig" in merged.columns:
        d["n_age_sig_col"] = int(merged["Age_sig"].astype(bool).sum())

    pair = merged[["Age_Beta", "Log2FC"]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(pair) > 2:
        rs, ps = spearmanr(pair["Age_Beta"], pair["Log2FC"])
        rp, pp = pearsonr(pair["Age_Beta"], pair["Log2FC"])
        d["spearman_r"] = float(rs)
        d["spearman_p"] = float(ps)
        d["pearson_r"] = float(rp)
        d["pearson_p"] = float(pp)
    else:
        d["spearman_r"] = d["spearman_p"] = d["pearson_r"] = d["pearson_p"] = float("nan")

    return d


def format_escape_merge_report(stats: dict[str, Any]) -> str:
    """Human-readable block for notebooks / logs."""
    lines = [f"=== {stats.get('label', 'merge')} ===", f"n_overlap: {stats.get('n_overlap', 0)}"]
    for k in (
        "n_ukbb_rows_input",
        "pct_ukbb_rows_in_overlap",
        "n_de_rows_input",
        "pct_de_rows_in_overlap",
    ):
        if k in stats and stats[k] == stats[k]:  # not NaN
            lines.append(f"{k}: {stats[k]:.4g}" if isinstance(stats[k], float) else f"{k}: {stats[k]}")
    if stats.get("n_overlap", 0) == 0:
        return "\n".join(lines)
    lines.extend(
        [
            f"Age_Beta: mean={stats.get('age_beta_mean'):.5g}  median={stats.get('age_beta_median'):.5g}  "
            f"pos/neg/zero={stats.get('age_beta_n_pos')}/{stats.get('age_beta_n_neg')}/{stats.get('age_beta_n_zero')}",
            f"log2FC:   mean={stats.get('log2fc_mean'):.5g}  median={stats.get('log2fc_median'):.5g}  "
            f"pos/neg/zero={stats.get('log2fc_n_pos')}/{stats.get('log2fc_n_neg')}/{stats.get('log2fc_n_zero')}",
        ]
    )
    for k in ("n_fdr_lt_005", "n_age_p_lt_005", "n_cent_sig", "n_age_sig_col"):
        if k in stats:
            lines.append(f"{k}: {stats[k]}")
    if not pd.isna(stats.get("spearman_r", float("nan"))):
        lines.append(
            f"Spearman(Age_Beta, log2FC): r={stats['spearman_r']:.4f}  p={stats['spearman_p']:.3g}"
        )
        lines.append(
            f"Pearson(Age_Beta, log2FC):  r={stats['pearson_r']:.4f}  p={stats['pearson_p']:.3g}"
        )
    return "\n".join(lines)


def print_escape_merge_report(stats: dict[str, Any]) -> None:
    print(format_escape_merge_report(stats))


def save_escape_merge_stats(
    rows: list[dict[str, Any]],
    out_path: str | Path,
) -> Path:
    """Save one or more ``summarize_escape_merge`` dicts as a wide CSV (one row per step)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path


def load_escape_gene_set(escape_master_path: str | Path | None) -> set[str]:
    """Genes with ``Is_Escape`` from optional ``escape_genes_master.csv`` (Figure 3)."""
    if escape_master_path is None:
        return set()
    p = Path(escape_master_path)
    if not p.is_file():
        return set()
    em = pd.read_csv(p)
    if "Is_Escape" not in em.columns or "Gene" not in em.columns:
        return set()
    return set(em.loc[em["Is_Escape"] == True, "Gene"].astype(str).str.strip().str.upper())


def build_bothsig_table(merged: pd.DataFrame) -> pd.DataFrame:
    """Subset both-significant rows + ``Concordant`` / ``Centenarian_Direction`` (``make_figure3_v3``)."""
    m = merged.copy()
    if "Log2FC" not in m.columns and "log2FC" in m.columns:
        m["Log2FC"] = pd.to_numeric(m["log2FC"], errors="coerce")
    bs = m[m["Cent_sig"] & m["Age_sig"]].copy()
    if bs.empty:
        return pd.DataFrame(
            columns=["Gene", "Concordant", "Age_direction", "Centenarian_Direction", "Log2FC", "Age_Beta"]
        )
    bs["Centenarian_Direction"] = np.where(
        bs["Log2FC"].astype(float) > 0,
        "increase",
        np.where(bs["Log2FC"].astype(float) < 0, "decrease", "flat"),
    )

    def _conc(r: pd.Series) -> bool:
        ad = r["Age_direction"]
        lfc = float(r["Log2FC"]) if pd.notna(r["Log2FC"]) else np.nan
        if pd.isna(lfc):
            return False
        if ad == "increase" and lfc > 0:
            return True
        if ad == "decrease" and lfc < 0:
            return True
        return False

    bs["Concordant"] = bs.apply(_conc, axis=1)
    bs = bs.drop_duplicates("Gene", keep="first")
    return bs[["Gene", "Concordant", "Age_direction", "Centenarian_Direction", "Log2FC", "Age_Beta"]]


def apply_escape_categories(
    df: pd.DataFrame,
    bothsig: pd.DataFrame | None = None,
    *,
    escape_set: set[str] | None = None,
) -> pd.DataFrame:
    """
    Apply ``classify`` from ``make_figure3_v3.py`` (lines 86–109).

    - Pass **only** the table you want labelled (e.g. ``ukbb_ciao_bothsig``). If ``bothsig`` is
      omitted, discordant/concordant reference rows are taken from :func:`build_bothsig_table`
      applied to ``df`` (correct when ``df`` is already the both-significant subset).
    - For the original Figure~3 workflow (full overlap + separate both-sig table), pass the full
      merged overlap as ``df`` and the output of :func:`build_bothsig_table` as ``bothsig``.

    ``escape_set`` can be empty; discordant both-significant hits still classify via the reference.
    """
    escape_set = escape_set or set()
    escape_set = {str(g).strip().upper() for g in escape_set}

    bothsig_ref = bothsig if bothsig is not None else build_bothsig_table(df)

    disc_bs = bothsig_ref[bothsig_ref["Concordant"] == False] if len(bothsig_ref) else pd.DataFrame()
    reversed_bs = disc_bs[
        (disc_bs["Age_direction"] == "decrease") & (disc_bs["Centenarian_Direction"] == "increase")
    ]
    reversed_set = set(reversed_bs["Gene"].astype(str).str.upper())

    def classify(row: pd.Series) -> str:
        c_sig = bool(row["Cent_sig"])
        a_sig = bool(row["Age_sig"])
        if not c_sig and not a_sig:
            return "NS"
        if a_sig and not c_sig:
            return "Aging only"
        if c_sig and not a_sig:
            return "Centenarian only"
        gene = str(row["Gene"]).strip().upper()
        if gene in escape_set:
            return "Escape"
        if gene in reversed_set:
            return "Reversed"
        bs_row = bothsig_ref[bothsig_ref["Gene"].astype(str).str.upper() == gene]
        if len(bs_row) > 0 and not bool(bs_row.iloc[0]["Concordant"]):
            age_dir = row["Age_direction"]
            log2fc = float(row["Log2FC"]) if pd.notna(row["Log2FC"]) else np.nan
            if age_dir == "increase" and log2fc < 0:
                return "Escape"
            if age_dir == "decrease" and log2fc > 0:
                return "Reversed"
        return "Concordant"

    out = df.copy()
    out["Category"] = out.apply(classify, axis=1)
    return out


def plot_escape_category_counts(
    merged: pd.DataFrame,
    out_path: str | Path | None = None,
    *,
    category_col: str = "Category",
    title: str = "UKBB × Centenarian overlap: category counts",
    show: bool = False,
) -> Path | None:
    order = ["NS", "Aging only", "Centenarian only", "Concordant", "Escape", "Reversed"]
    vc = merged[category_col].value_counts()
    counts = [int(vc.get(c, 0)) for c in order]
    labels = [f"{c}\n(n={counts[i]})" for i, c in enumerate(order)]
    colors = [ESCAPE_CATEGORY_COLORS[c] for c in order]

    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(order))
    ax.bar(x, counts, color=colors, edgecolor="white", linewidth=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Proteins", fontsize=10)
    ax.set_title(title, fontsize=11)
    sns.despine(ax=ax)
    plt.tight_layout()
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    plt.close(fig)
    return Path(out_path) if out_path is not None else None


# Drawn under classified points when plotting full overlap + both-sig categories only.
NOT_BOTHSIG_CATEGORY = "Not both-sig"


def _plot_age_beta_vs_log2fc_scatter_r(
    merged: pd.DataFrame,
    out_path: Path,
    *,
    category_col: str,
    label_top_escape: int,
    label_top_reversed: int,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    """Render scatter with ggplot2 + ggrepel via rpy2 (requires R + ggplot2 + ggrepel)."""
    import rpy2.robjects as ro
    from rpy2.robjects.vectors import FloatVector, IntVector, StrVector

    C = ESCAPE_CATEGORY_COLORS
    cat_config: dict[str, dict[str, Any]] = {
        NOT_BOTHSIG_CATEGORY: {"color": C["NS"], "shape": 16, "size": 1.55},
        "NS": {"color": C["NS"], "shape": 16, "size": 1.65},
        "Aging only": {"color": C["Aging only"], "shape": 16, "size": 1.85},
        "Centenarian only": {"color": C["Centenarian only"], "shape": 16, "size": 1.85},
        "Concordant": {"color": C["Concordant"], "shape": 16, "size": 2.15},
        "Escape": {"color": C["Escape"], "shape": 15, "size": 2.75},
        "Reversed": {"color": C["Reversed"], "shape": 17, "size": 2.85},
    }

    plot_order = [
        NOT_BOTHSIG_CATEGORY,
        "NS",
        "Aging only",
        "Centenarian only",
        "Concordant",
        "Escape",
        "Reversed",
    ]

    need = {"Gene", "Age_Beta", "Log2FC", category_col}
    missing = need - set(merged.columns)
    if missing:
        raise ValueError(f"merged is missing columns: {sorted(missing)}")

    plot_df = merged[list(need)].copy()
    plot_df["Age_Beta"] = pd.to_numeric(plot_df["Age_Beta"], errors="coerce")
    plot_df["Log2FC"] = pd.to_numeric(plot_df["Log2FC"], errors="coerce")
    plot_df = plot_df[np.isfinite(plot_df["Age_Beta"]) & np.isfinite(plot_df["Log2FC"])].copy()
    plot_df = plot_df.rename(columns={category_col: "PlotCategory"})

    label_rows: list[dict[str, Any]] = []
    esc = plot_df[plot_df["PlotCategory"] == "Escape"].copy()
    rev = plot_df[plot_df["PlotCategory"] == "Reversed"].copy()
    if len(esc) and label_top_escape > 0:
        esc = esc.assign(_a=esc["Log2FC"].astype(float).abs()).sort_values("_a", ascending=False).head(label_top_escape)
        for _, r in esc.iterrows():
            label_rows.append(
                {
                    "Gene": r["Gene"],
                    "Age_Beta": r["Age_Beta"],
                    "Log2FC": r["Log2FC"],
                    "grp": "escape",
                }
            )
    if len(rev) and label_top_reversed > 0:
        rev = rev.assign(_a=rev["Log2FC"].astype(float).abs()).sort_values("_a", ascending=False).head(label_top_reversed)
        for _, r in rev.iterrows():
            label_rows.append(
                {
                    "Gene": r["Gene"],
                    "Age_Beta": r["Age_Beta"],
                    "Log2FC": r["Log2FC"],
                    "grp": "reversed",
                }
            )
    labels_df = pd.DataFrame(label_rows)

    with tempfile.TemporaryDirectory() as td:
        tdir = Path(td)
        csv_main = tdir / "merged.csv"
        csv_lab = tdir / "labels.csv"
        plot_df.to_csv(csv_main, index=False)
        if len(labels_df):
            labels_df.to_csv(csv_lab, index=False)
        else:
            labels_df = pd.DataFrame(columns=["Gene", "Age_Beta", "Log2FC", "grp"])
            labels_df.to_csv(csv_lab, index=False)

        ro.globalenv["csv_main"] = str(csv_main.resolve())
        ro.globalenv["csv_lab"] = str(csv_lab.resolve())
        ro.globalenv["out_file"] = str(out_path.resolve())
        ro.globalenv["main_title"] = title
        ro.globalenv["xlab"] = xlabel
        ro.globalenv["ylab"] = ylabel

        lv = [c for c in plot_order if c in cat_config]
        ro.globalenv["plot_levels"] = StrVector(lv)
        ro.globalenv["col_vals"] = StrVector([cat_config[c]["color"] for c in lv])
        ro.globalenv["shp_vals"] = IntVector([int(cat_config[c]["shape"]) for c in lv])
        ro.globalenv["size_vals"] = FloatVector([float(cat_config[c]["size"]) for c in lv])
        ro.globalenv["alpha_vals"] = FloatVector([0.28, 0.35, 0.5, 0.5, 0.58, 0.88, 0.88][: len(lv)])

        ro.r(
            """
            suppressPackageStartupMessages({
              library(ggplot2)
              library(ggrepel)
            })
            d <- read.csv(csv_main, stringsAsFactors = FALSE, check.names = FALSE)
            if (!"PlotCategory" %in% names(d)) stop("PlotCategory column missing in CSV")
            d[["PlotCategory"]] <- factor(d[["PlotCategory"]], levels = plot_levels)
            lab <- read.csv(csv_lab, stringsAsFactors = FALSE, check.names = FALSE)

            cols <- col_vals
            names(cols) <- plot_levels
            shps <- shp_vals
            names(shps) <- plot_levels
            szs <- size_vals
            names(szs) <- plot_levels
            alfs <- alpha_vals
            names(alfs) <- plot_levels

            p <- ggplot(d, aes(
                x = Age_Beta, y = Log2FC,
                colour = PlotCategory, shape = PlotCategory,
                size = PlotCategory, alpha = PlotCategory
              )) +
              geom_hline(yintercept = 0, colour = "#7A7876", linewidth = 0.35, linetype = "dashed") +
              geom_vline(xintercept = 0, colour = "#7A7876", linewidth = 0.35, linetype = "dashed") +
              geom_point() +
              scale_colour_manual(values = cols, drop = FALSE) +
              scale_shape_manual(values = shps, drop = FALSE) +
              scale_size_manual(values = szs, drop = FALSE) +
              scale_alpha_manual(values = alfs, drop = FALSE) +
              labs(title = main_title, x = xlab, y = ylab) +
              theme_bw(base_size = 11) +
              theme(
                legend.position = "right",
                legend.title = element_blank(),
                plot.title = element_text(face = "bold", hjust = 0.5),
                panel.grid.minor = element_blank()
              ) +
              guides(
                colour = guide_legend(ncol = 1, override.aes = list(size = 3, alpha = 1)),
                shape = "none",
                size = "none",
                alpha = "none"
              )

            if (nrow(lab) > 0 && all(c("Gene", "Age_Beta", "Log2FC", "grp") %in% names(lab))) {
              le <- lab[lab$grp == "escape", , drop = FALSE]
              lr <- lab[lab$grp == "reversed", , drop = FALSE]
              if (nrow(le) > 0) {
                p <- p + geom_text_repel(
                  data = le,
                  aes(x = Age_Beta, y = Log2FC, label = Gene),
                  colour = "#2d2d2d",
                  inherit.aes = FALSE,
                  fontface = "bold.italic",
                  size = 3.0,
                  min.segment.length = 0,
                  max.overlaps = Inf,
                  force = 1.8,
                  force_pull = 0.85,
                  box.padding = 0.55,
                  point.padding = 0.4,
                  segment.color = "#333333",
                  segment.size = 0.25,
                  show.legend = FALSE
                )
              }
              if (nrow(lr) > 0) {
                p <- p + geom_text_repel(
                  data = lr,
                  aes(x = Age_Beta, y = Log2FC, label = Gene),
                  colour = "#1f3d36",
                  inherit.aes = FALSE,
                  fontface = "bold.italic",
                  size = 3.0,
                  min.segment.length = 0,
                  max.overlaps = Inf,
                  force = 1.8,
                  force_pull = 0.85,
                  box.padding = 0.55,
                  point.padding = 0.4,
                  segment.color = "#333333",
                  segment.size = 0.25,
                  show.legend = FALSE
                )
              }
            }

            w_in <- 8.5
            h_in <- 6.5
            ggsave(
              filename = out_file,
              plot = p,
              width = w_in,
              height = h_in,
              dpi = 300,
              bg = "white"
            )
            """
        )


def _plot_age_beta_vs_log2fc_scatter_matplotlib(
    merged: pd.DataFrame,
    out_path: Path | None,
    *,
    category_col: str,
    label_top_escape: int,
    label_top_reversed: int,
    title: str,
    xlabel: str,
    ylabel: str,
    show: bool,
) -> Path | None:
    """Matplotlib + adjustText fallback."""
    C = ESCAPE_CATEGORY_COLORS
    # Marker sizes / alphas roughly scaled from make_figure3_v3 panel a (matplotlib s ≈ 4× area in pt²).
    cat_config: dict[str, dict[str, Any]] = {
        NOT_BOTHSIG_CATEGORY: {
            "color": C["NS"],
            "marker": "o",
            "s": 22,
            "alpha": 0.28,
            "zorder": 0,
            "edge": "none",
        },
        "NS": {"color": C["NS"], "marker": "o", "s": 28, "alpha": 0.35, "zorder": 1, "edge": "none"},
        "Aging only": {"color": C["Aging only"], "marker": "o", "s": 36, "alpha": 0.5, "zorder": 2, "edge": "none"},
        "Centenarian only": {
            "color": C["Centenarian only"],
            "marker": "o",
            "s": 36,
            "alpha": 0.5,
            "zorder": 2,
            "edge": "none",
        },
        "Concordant": {"color": C["Concordant"], "marker": "o", "s": 52, "alpha": 0.58, "zorder": 3, "edge": "none"},
        "Escape": {"color": C["Escape"], "marker": "s", "s": 95, "alpha": 0.88, "zorder": 5, "edge": "highlight"},
        "Reversed": {"color": C["Reversed"], "marker": "^", "s": 105, "alpha": 0.88, "zorder": 5, "edge": "highlight"},
    }

    plot_order = [
        NOT_BOTHSIG_CATEGORY,
        "NS",
        "Aging only",
        "Centenarian only",
        "Concordant",
        "Escape",
        "Reversed",
    ]

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    for cat in plot_order:
        if cat not in cat_config:
            continue
        sub = merged[merged[category_col] == cat]
        if sub.empty:
            continue
        cfg = cat_config[cat]
        edge = "white" if cfg.get("edge") == "highlight" else "none"
        lw = 0.35 if edge != "none" else 0.0
        leg = f"{cat} (overlap, not both-sig)" if cat == NOT_BOTHSIG_CATEGORY else cat
        ax.scatter(
            sub["Age_Beta"],
            sub["Log2FC"],
            c=cfg["color"],
            marker=cfg["marker"],
            s=cfg["s"],
            alpha=cfg["alpha"],
            zorder=cfg["zorder"],
            edgecolors=edge,
            linewidths=lw,
            label=leg,
        )

    ax.axhline(0, color="#7A7876", lw=0.55, ls="--", zorder=0)
    ax.axvline(0, color="#7A7876", lw=0.55, ls="--", zorder=0)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.legend(frameon=False, loc="best", fontsize=8, ncol=2)
    sns.despine(ax=ax)
    plt.tight_layout()

    esc = merged[merged[category_col] == "Escape"].copy()
    rev = merged[merged[category_col] == "Reversed"].copy()
    texts = []
    _halo = [mpe.withStroke(linewidth=3.0, foreground="white", capstyle="round", joinstyle="round")]

    if len(esc) and label_top_escape > 0:
        esc = esc.assign(_a=esc["Log2FC"].astype(float).abs()).sort_values("_a", ascending=False).head(label_top_escape)
        for _, r in esc.iterrows():
            t = ax.annotate(
                r["Gene"],
                (r["Age_Beta"], r["Log2FC"]),
                fontsize=8.5,
                fontweight="bold",
                fontstyle="italic",
                color="#2d2d2d",
                ha="center",
                va="center",
                zorder=20,
            )
            t.set_path_effects(_halo)
            texts.append(t)
    if len(rev) and label_top_reversed > 0:
        rev = rev.assign(_a=rev["Log2FC"].astype(float).abs()).sort_values("_a", ascending=False).head(label_top_reversed)
        for _, r in rev.iterrows():
            t = ax.annotate(
                r["Gene"],
                (r["Age_Beta"], r["Log2FC"]),
                fontsize=8.5,
                fontweight="bold",
                fontstyle="italic",
                color="#1f3d36",
                ha="center",
                va="center",
                zorder=20,
            )
            t.set_path_effects(_halo)
            texts.append(t)

    try:
        from adjustText import adjust_text

        if texts:
            sx = pd.to_numeric(merged["Age_Beta"], errors="coerce").to_numpy(dtype=float)
            sy = pd.to_numeric(merged["Log2FC"], errors="coerce").to_numpy(dtype=float)
            ok = np.isfinite(sx) & np.isfinite(sy)
            sx, sy = sx[ok], sy[ok]
            ap = dict(arrowstyle="-", color="#333333", lw=0.35, shrinkA=0, shrinkB=2)
            sig = inspect.signature(adjust_text)
            params = sig.parameters

            if "iter_lim" in params:
                # adjustText 1.x — lim / force_points / expand_* are ignored; use iter_lim + force_static.
                adjust_kw: dict[str, Any] = dict(
                    ax=ax,
                    arrowprops=ap,
                    force_text=(1.05, 1.2),
                    force_static=(0.62, 0.72),
                    force_explode=(0.52, 1.05),
                    force_pull=(0.02, 0.024),
                    expand=(1.72, 1.95),
                    max_move=(16, 16),
                    iter_lim=6000,
                    prevent_crossings=True,
                )
            else:
                # adjustText 0.7.x
                adjust_kw = dict(
                    ax=ax,
                    arrowprops=ap,
                    force_text=(1.35, 1.55),
                    force_points=(0.85, 0.95),
                    expand_text=(1.65, 1.88),
                    expand_points=(1.48, 1.68),
                    lim=4500,
                )
            if len(sx) > 0:
                adjust_text(texts, x=sx, y=sy, **adjust_kw)
            else:
                adjust_text(texts, **adjust_kw)
    except Exception:
        pass
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    plt.close(fig)
    return Path(out_path) if out_path is not None else None


def plot_age_beta_vs_log2fc_scatter(
    merged: pd.DataFrame,
    out_path: str | Path | None = None,
    *,
    category_col: str = "Category",
    label_top_escape: int = 8,
    label_top_reversed: int = 8,
    title: str = "UKBB Age Beta vs Centenarian log2FC (colored by category)",
    show: bool = False,
    use_r: bool = True,
    matplotlib_fallback: bool = True,
    **kwargs: Any,
) -> Path | None:
    """
    Scatter ``Age_Beta`` vs ``Log2FC`` with categories coloured; label top Escape / Reversed genes.

    By default uses **ggplot2 + ggrepel** via **rpy2** (requires a working R install with
    ``ggplot2`` and ``ggrepel``). Falls back to matplotlib + adjustText if ``use_r`` is
    False, or if R fails and ``matplotlib_fallback`` is True.

    Optional keyword arguments: ``xlabel``, ``ylabel``.
    """
    xlabel = kwargs.pop("xlabel", "UKBB Age Beta")
    ylabel = kwargs.pop("ylabel", "Centenarian log2FC (Centenarian − Control)")
    if kwargs:
        raise TypeError(f"plot_age_beta_vs_log2fc_scatter: unexpected keyword arguments {set(kwargs)!r}")

    resolved_out = Path(out_path) if out_path is not None else None

    if use_r and resolved_out is not None:
        try:
            resolved_out.parent.mkdir(parents=True, exist_ok=True)
            _plot_age_beta_vs_log2fc_scatter_r(
                merged,
                resolved_out,
                category_col=category_col,
                label_top_escape=label_top_escape,
                label_top_reversed=label_top_reversed,
                title=title,
                xlabel=xlabel,
                ylabel=ylabel,
            )
            if show:
                try:
                    from IPython.display import Image, display

                    display(Image(filename=str(resolved_out)))
                except Exception:
                    pass
            return resolved_out
        except Exception:
            if not matplotlib_fallback:
                raise

    return _plot_age_beta_vs_log2fc_scatter_matplotlib(
        merged,
        resolved_out,
        category_col=category_col,
        label_top_escape=label_top_escape,
        label_top_reversed=label_top_reversed,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        show=show,
    )

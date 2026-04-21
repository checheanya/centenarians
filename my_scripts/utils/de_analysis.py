"""Differential expression: Student's t, FDR, volcano (with labels), LFC barplot, GO/KEGG enrichment."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests

# Enrichr library names (gseapy / Enrichr)
GO_BIOLOGICAL_PROCESS = "GO_Biological_Process_2023"
KEGG_HUMAN = "KEGG_2021_Human"


def differential_expression_student_t(
    X_log2_norm: pd.DataFrame,
    id_df: pd.DataFrame,
    sample_to_group: dict[str, str],
    *,
    group_numerator: str = "Centenarian",
    group_denominator: str = "Control",
    min_per_group: int = 2,
) -> pd.DataFrame:
    """
    Two-sided **Student's t-test** (equal variance) per protein.

    log2FC = mean(log2 abundance in numerator group) − mean(in denominator group).
    Positive log2FC ⇒ higher in numerator (e.g. Centenarian vs Control).

    Benjamini–Hochberg FDR on raw p-values.
    """
    g = pd.Series(sample_to_group)
    cols = g.index.intersection(X_log2_norm.columns)
    g = g.loc[cols]
    s_num = g[g == group_numerator].index.tolist()
    s_den = g[g == group_denominator].index.tolist()
    if len(s_num) < min_per_group or len(s_den) < min_per_group:
        raise ValueError("Not enough samples per group for DE.")

    mat = X_log2_norm[cols]
    log2fc: list[float] = []
    pvals: list[float] = []
    for _, row in mat.iterrows():
        a = row[s_num].dropna().values.astype(float)
        b = row[s_den].dropna().values.astype(float)
        if len(a) < min_per_group or len(b) < min_per_group:
            log2fc.append(np.nan)
            pvals.append(np.nan)
            continue
        log2fc.append(float(np.nanmean(a) - np.nanmean(b)))
        _, p = stats.ttest_ind(a, b, equal_var=True)
        pvals.append(float(p))

    res = id_df.copy()
    res["log2FC"] = log2fc
    res["pvalue"] = pvals
    mask = res["pvalue"].notna()
    if mask.any():
        res.loc[mask, "padj"] = multipletests(res.loc[mask, "pvalue"], method="fdr_bh")[1]
    else:
        res["padj"] = np.nan
    return res


def summarize_de_results(
    res: pd.DataFrame,
    *,
    fdr_thresh: float = 0.05,
    fc_thresh: float = 0.5,
) -> str:
    """
    Printable summary: proteins tested, FDR hits, up/down counts, strongest hit.

    Uses the same FDR / |log2FC| thresholds as the volcano and barplots.
    """
    df = res.dropna(subset=["pvalue"])
    n = len(df)
    if n == 0:
        return "No proteins with valid p-values."
    padj_ok = df["padj"].notna()
    n_fdr_lt = int((df.loc[padj_ok, "padj"] < fdr_thresh).sum())
    sig = df.loc[padj_ok & (df["padj"] < fdr_thresh)]
    n_up = int((sig["log2FC"] > fc_thresh).sum())
    n_down = int((sig["log2FC"] < -fc_thresh).sum())
    n_mid = int((sig["log2FC"].abs() <= fc_thresh).sum())
    lines = [
        f"Proteins tested (non-null p-value): {n}",
        f"FDR < {fdr_thresh}: {n_fdr_lt}",
        f"  — with log2FC > +{fc_thresh} (up in Centenarian): {n_up}",
        f"  — with log2FC < −{fc_thresh} (down in Centenarian): {n_down}",
        f"  — significant but |log2FC| ≤ {fc_thresh}: {n_mid}",
    ]
    sub = df.loc[df["padj"].notna()].sort_values("padj", ascending=True)
    if len(sub) > 0:
        br = sub.iloc[0]
        gdf = sub.iloc[[0]]
        g = _gene_symbol_series(gdf).iloc[0] or "(no symbol)"
        lines.append(
            f"Strongest FDR: padj={float(br['padj']):.3e}, log2FC={float(br['log2FC']):.3f}, gene={g}"
        )
    return "\n".join(lines)


def _volcano_colors(
    log2fc: pd.Series,
    padj: pd.Series,
    *,
    fdr_thresh: float = 0.05,
    fc_thresh: float = 0.5,
) -> list[str]:
    colors: list[str] = []
    for fc, q in zip(log2fc, padj):
        if q is None or (isinstance(q, float) and np.isnan(q)) or q >= fdr_thresh:
            colors.append("#d9d9d9")
        elif abs(fc) <= fc_thresh:
            colors.append("#7f7f7f")
        elif fc > fc_thresh:
            colors.append("#c0392b")
        else:
            colors.append("#2980b9")
    return colors


def _gene_symbol_series(df: pd.DataFrame, gene_col: str = "PG.Genes") -> pd.Series:
    if "gene_symbol" in df.columns:
        return df["gene_symbol"].fillna("").astype(str).str.strip()
    if gene_col in df.columns:
        return df[gene_col].astype(str).str.split(";").str[0].str.strip()
    return pd.Series([""] * len(df), index=df.index)


def plot_volcano(
    res: pd.DataFrame,
    out_path: str | Path,
    *,
    fdr_thresh: float = 0.05,
    fc_thresh: float = 0.5,
    x_col: str = "log2FC",
    title: str = "Volcano (Centenarian vs Control)",
    figsize: tuple[float, float] = (8, 7),
    label_top_n: int = 20,
    gene_col: str = "PG.Genes",
    show: bool = False,
) -> Path:
    """Volcano: x = log2FC, y = -log10(p). Labels the ``label_top_n`` smallest *padj* (most significant)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = res.dropna(subset=["pvalue", x_col]).copy()
    df["neglog10p"] = -np.log10(df["pvalue"].clip(lower=1e-300))
    df["color"] = _volcano_colors(df[x_col], df["padj"], fdr_thresh=fdr_thresh, fc_thresh=fc_thresh)
    df["_gsym"] = _gene_symbol_series(df, gene_col=gene_col)

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(
        df[x_col],
        df["neglog10p"],
        c=df["color"],
        s=22,
        alpha=0.75,
        edgecolors="none",
    )
    ax.axvline(-fc_thresh, color="0.5", ls="--", lw=0.6)
    ax.axvline(fc_thresh, color="0.5", ls="--", lw=0.6)
    ax.set_xlabel("log2 FC (Centenarian − Control)")
    ax.set_ylabel(r"$-\log_{10}(p)$")
    ax.set_title(title)
    from matplotlib.patches import Patch

    legend_elems = [
        Patch(facecolor="#c0392b", label=f"FDR<{fdr_thresh} & FC>{fc_thresh}"),
        Patch(facecolor="#2980b9", label=f"FDR<{fdr_thresh} & FC<−{fc_thresh}"),
        Patch(facecolor="#7f7f7f", label=f"FDR<{fdr_thresh}, |FC|≤{fc_thresh}"),
        Patch(facecolor="#d9d9d9", label=f"FDR≥{fdr_thresh}"),
    ]
    ax.legend(handles=legend_elems, frameon=False, loc="upper right", fontsize=8)

    if label_top_n > 0:
        lab = df.dropna(subset=["padj"]).sort_values("padj", ascending=True).head(label_top_n)
        lab = lab[lab["_gsym"].astype(str).str.len() > 0]
        texts: list[Any] = []
        for _, row in lab.iterrows():
            t = ax.text(
                row[x_col],
                row["neglog10p"],
                str(row["_gsym"]),
                fontsize=7,
                ha="center",
                va="bottom",
                clip_on=False,
            )
            texts.append(t)
        try:
            from adjustText import adjust_text

            adjust_text(
                texts,
                ax=ax,
                arrowprops=dict(arrowstyle="-", color="0.35", lw=0.4, shrinkA=2, shrinkB=2),
            )
        except ImportError:
            pass

    sns.despine(ax=ax)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def build_top_de_table(
    res: pd.DataFrame,
    *,
    top_n_per_direction: int = 10,
    fdr_thresh: float = 0.05,
    fc_thresh: float = 0.5,
    gene_col: str = "PG.Genes",
) -> pd.DataFrame:
    """
    Up to ``top_n_per_direction`` elevated and ``top_n_per_direction`` reduced among
    ``padj < fdr_thresh`` and |log2FC| > ``fc_thresh``, ranked by log2FC.
    """
    sig = res.dropna(subset=["padj", "log2FC"])
    sig = sig[sig["padj"] < fdr_thresh]
    up = sig[sig["log2FC"] > fc_thresh].sort_values("log2FC", ascending=False).head(top_n_per_direction)
    down = sig[sig["log2FC"] < -fc_thresh].sort_values("log2FC", ascending=True).head(top_n_per_direction)
    up = up.assign(direction="elevated_in_Centenarian")
    down = down.assign(direction="reduced_in_Centenarian")
    out = pd.concat([up, down], ignore_index=True)
    if gene_col in out.columns:
        out["gene_symbol"] = out[gene_col].astype(str).str.split(";").str[0]
    return out


def build_top30_table(
    res: pd.DataFrame,
    *,
    fdr_thresh: float = 0.05,
    fc_thresh: float = 0.5,
    gene_col: str = "PG.Genes",
) -> pd.DataFrame:
    """Backward-compatible alias: 15 up + 15 down (legacy). Prefer ``build_top_de_table``."""
    return build_top_de_table(
        res,
        top_n_per_direction=15,
        fdr_thresh=fdr_thresh,
        fc_thresh=fc_thresh,
        gene_col=gene_col,
    )


def plot_top_lfc_barplot(
    res: pd.DataFrame,
    out_path: str | Path,
    *,
    top_n: int = 10,
    fdr_thresh: float = 0.05,
    fc_thresh: float = 0.5,
    gene_col: str = "PG.Genes",
    title: str = "Top DE proteins by log2FC",
    show: bool = False,
) -> Path:
    """Single horizontal bar chart: top ``top_n`` up and top ``top_n`` down on one y-axis (sorted by log2FC)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sig = res.dropna(subset=["padj", "log2FC"])
    sig = sig[sig["padj"] < fdr_thresh]
    gsym = _gene_symbol_series(sig, gene_col=gene_col)
    sig = sig.assign(_gsym=gsym)

    up = sig[sig["log2FC"] > fc_thresh].sort_values("log2FC", ascending=False).head(top_n)
    down = sig[sig["log2FC"] < -fc_thresh].sort_values("log2FC", ascending=True).head(top_n)

    combined = pd.concat([up, down])
    if combined.empty:
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.text(0.5, 0.5, "No proteins pass FDR / FC thresholds", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        plt.tight_layout()
        fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
        if show:
            plt.show()
        plt.close(fig)
        return out_path

    # Highest log2FC at top of figure (invert_yaxis → first row on top)
    combined = combined.sort_values("log2FC", ascending=False)
    names = combined["_gsym"].astype(str).where(
        combined["_gsym"].astype(str).str.len() > 0,
        combined.index.astype(str),
    )
    vals = combined["log2FC"].astype(float).values
    colors = np.where(vals > 0, "#c0392b", "#2980b9")
    y = np.arange(len(combined))

    fig_h = max(4.0, 0.38 * len(combined) + 1.6)
    fig, ax = plt.subplots(figsize=(8, fig_h))
    ax.barh(y, vals, color=colors, edgecolor="white", linewidth=0.4, height=0.72)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.axvline(0, color="0.35", lw=0.7)
    ax.set_xlabel("log2 FC (Centenarian − Control)")
    ax.set_title(title, fontsize=11)
    ax.invert_yaxis()
    span = max(float(np.nanmax(np.abs(vals))), 0.05)
    pad = 0.06 * span
    for yi, v in zip(y, vals):
        x_text = v + pad if v >= 0 else v - pad
        ax.text(x_text, yi, f"{v:.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=7)
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(facecolor="#c0392b", label=f"Elevated (n≤{top_n})"),
            Patch(facecolor="#2980b9", label=f"Reduced (n≤{top_n})"),
        ],
        frameon=False,
        loc="lower right",
        fontsize=8,
    )
    sns.despine(ax=ax)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def _parse_enrichr_overlap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrichr ``Overlap`` is ``k/n`` (genes in query ∩ set / genes in set). Add explicit columns.
    """
    out = df.copy()
    if "Overlap" not in out.columns:
        out["overlap_genes"] = np.nan
        out["gene_set_size"] = np.nan
        return out

    def _one(val: object) -> tuple[float, float]:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return np.nan, np.nan
        s = str(val).strip()
        if "/" not in s:
            return np.nan, np.nan
        parts = s.split("/", 1)
        try:
            k, n = int(parts[0].strip()), int(parts[1].strip())
            return float(k), float(n)
        except (ValueError, IndexError):
            return np.nan, np.nan

    k_n = out["Overlap"].map(_one)
    out["overlap_genes"] = k_n.map(lambda t: t[0])
    out["gene_set_size"] = k_n.map(lambda t: t[1])
    return out


def _enrichr_dotplot_gene_set_sizes(
    df: pd.DataFrame,
    out_plot: Path,
    *,
    adj_col: str = "Adjusted P-value",
    fdr_max: float = 0.25,
    top_n: int = 15,
    plot_title: str = "",
    show: bool = False,
    figsize: tuple[float, float] | None = None,
) -> bool:
    """
    Dot plot: x = −log10(adj. P), y = term, **dot area ∝ gene set size** (library ``n`` from Overlap),
    color = −log10(adj. P). Includes size legend for gene-set size (gseapy-style).
    """
    if adj_col not in df.columns:
        return False
    sub = df.dropna(subset=[adj_col]).copy()
    sub[adj_col] = sub[adj_col].astype(float)
    sub = sub[sub[adj_col] <= fdr_max]
    if len(sub) < 1:
        return False
    sub = sub.sort_values(adj_col, ascending=True).head(int(top_n))
    sub = _parse_enrichr_overlap(sub)
    sub["neglog10_padj"] = -np.log10(np.clip(sub[adj_col].astype(float), 1e-300, None))
    gs = sub["gene_set_size"]
    if gs.isna().all():
        gs = sub.get("overlap_genes", pd.Series(np.nan, index=sub.index))
    sizes = gs.astype(float)
    if sizes.isna().all():
        sizes = pd.Series(100.0, index=sub.index)
    else:
        sizes = sizes.fillna(np.nanmedian(sizes))

    terms = sub["Term"].astype(str).map(lambda t: (t[:72] + "…") if len(t) > 72 else t)
    h = max(5.0, 0.42 * len(sub) + 1.8)
    fs = figsize if figsize is not None else (10.0, h)
    fig, ax = plt.subplots(figsize=fs)
    x = sub["neglog10_padj"].values
    y = np.arange(len(sub))
    # Map gene_set_size to point area (similar spirit to gseapy DotPlot)
    s_min, s_max = float(np.nanmin(sizes)), float(np.nanmax(sizes))
    if s_max <= s_min:
        area = np.full(len(sub), 120.0)
    else:
        area = 40.0 + 280.0 * (sizes.values - s_min) / (s_max - s_min)

    sc = ax.scatter(
        x,
        y,
        s=area,
        c=sub["neglog10_padj"].values,
        cmap="viridis_r",
        alpha=0.92,
        edgecolors="0.35",
        linewidths=0.6,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(terms, fontsize=8)
    ax.set_xlabel(r"$-\log_{10}$(adj. P-value)", fontsize=10)
    ax.set_ylabel("")
    ax.set_title(plot_title or "Enrichr (dot size ∝ gene set size)", fontsize=11)
    ax.invert_yaxis()
    plt.colorbar(sc, ax=ax, shrink=0.55, label=r"$-\log_{10}$(adj. P)")
    # Size legend (representative gene set sizes)
    if s_max > s_min:
        leg_sizes = np.quantile(sizes.values, [0.25, 0.5, 0.75, 1.0])
        leg_sizes = np.unique(np.round(leg_sizes).astype(int))
        leg_sizes = leg_sizes[leg_sizes > 0][:4]
        handles = []
        from matplotlib.lines import Line2D

        for sz in leg_sizes:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="0.45",
                    markeredgecolor="0.35",
                    markersize=np.sqrt(40.0 + 280.0 * (float(sz) - s_min) / (s_max - s_min)) / 2.2,
                    linestyle="None",
                    label=str(int(sz)),
                )
            )
        ax.legend(
            handles,
            [str(int(s)) for s in leg_sizes],
            title="Gene set size",
            loc="lower right",
            frameon=True,
            fontsize=7,
            title_fontsize=8,
        )
    plt.tight_layout()
    out_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_plot, dpi=300, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    plt.close(fig)
    return True


def _prerank_nes_barplot(
    df: pd.DataFrame,
    out_path: Path,
    *,
    col_fdr: str = "FDR q-val",
    col_nes: str = "NES",
    fdr_cutoff: float | None = 0.25,
    top_n: int = 25,
    title: str = "",
    show: bool = False,
) -> bool:
    if col_nes not in df.columns:
        return False
    sub = df.copy()
    if fdr_cutoff is not None and col_fdr in sub.columns:
        sub[col_fdr] = sub[col_fdr].astype(float)
        sub = sub[sub[col_fdr] <= fdr_cutoff]
    if len(sub) < 1:
        return False
    sub = sub.reindex(sub[col_nes].abs().sort_values(ascending=False).index).head(int(top_n))
    terms = sub["Term"].astype(str).map(lambda t: (t[:70] + "…") if len(t) > 70 else t)
    nes = sub[col_nes].astype(float).values
    colors = np.where(nes >= 0, "#c0392b", "#2980b9")
    y = np.arange(len(sub))
    fig_h = max(4.5, 0.35 * len(sub) + 1.5)
    fig, ax = plt.subplots(figsize=(8.5, fig_h))
    ax.barh(y, nes, color=colors, edgecolor="white", linewidth=0.35, height=0.72)
    ax.set_yticks(y)
    ax.set_yticklabels(terms, fontsize=8)
    ax.axvline(0, color="0.4", lw=0.8)
    ax.set_xlabel("Normalized enrichment score (NES)", fontsize=10)
    ax.set_title(title or "Prerank GSEA — NES", fontsize=11)
    ax.invert_yaxis()
    sns.despine(ax=ax)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    plt.close(fig)
    return True


def _prerank_gseaplot_classic(
    pre: Any,
    out_dir: Path,
    *,
    stem: str,
    top_n: int = 3,
    show: bool = False,
) -> int:
    """Classical GSEA running-enrichment plots (``gseapy.plot.gseaplot``) for top pathways by |NES|."""
    try:
        from gseapy.plot import gseaplot
    except ImportError:
        return 0
    rnk = getattr(pre, "ranking", None)
    if rnk is None:
        return 0
    if isinstance(rnk, pd.DataFrame):
        rank_metric = rnk.iloc[:, 0].values
    else:
        rank_metric = np.asarray(rnk.values, dtype=float)

    res_dict = pre.results
    if not isinstance(res_dict, dict) or not res_dict:
        return 0
    sample = next(iter(res_dict.values()))
    if isinstance(sample, dict) and "hits" in sample:
        term_map: dict[str, Any] = res_dict  # type: ignore[assignment]
    else:
        lib0 = next(iter(res_dict.keys()))
        term_map = res_dict[lib0]
        if not isinstance(term_map, dict):
            return 0

    df = getattr(pre, "res2d", None)
    if df is None or len(df) == 0:
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    n_saved = 0
    sub = df.reindex(df["NES"].abs().sort_values(ascending=False).index).head(int(top_n))
    for _, row in sub.iterrows():
        term = str(row["Term"])
        if term not in term_map:
            continue
        rec = term_map[term]
        hits = rec.get("hits")
        res_arr = rec.get("RES")
        if hits is None or res_arr is None:
            continue
        hits = np.asarray(hits)
        res_arr = np.asarray(res_arr)
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in term.split("__")[-1])[:60]
        ofn = out_dir / f"{stem}_gsea_{n_saved + 1:02d}_{safe}.png"
        try:
            gseaplot(
                term=term.split("__")[-1],
                hits=hits,
                nes=float(row["NES"]),
                pval=float(row["NOM p-val"]),
                fdr=float(row["FDR q-val"]),
                RES=res_arr,
                rank_metric=rank_metric,
                pheno_pos=getattr(pre, "pheno_pos", "Pos"),
                pheno_neg=getattr(pre, "pheno_neg", "Neg"),
                figsize=(6.5, 6.0),
                ofname=str(ofn),
            )
            n_saved += 1
        except Exception:
            continue
    return n_saved


def _de_to_prerank_dataframe(
    res: pd.DataFrame,
    *,
    gene_col: str = "PG.Genes",
    rank_stat: Literal["log2fc", "log2fc_neglog10p", "signed_neglog10p"] = "log2fc_neglog10p",
) -> pd.DataFrame:
    """
    Two-column ranking table for ``gseapy.prerank``: gene symbol, score (higher = more ranked first).

    ``log2fc_neglog10p`` uses log2FC × −log10(p) (magnitude of differential signal).
    """
    df = res.dropna(subset=["pvalue", "log2FC"]).copy()
    g = _gene_symbol_series(df, gene_col=gene_col)
    df["_g"] = g.astype(str).str.strip()
    df = df[(df["_g"].str.len() > 0) & (df["_g"].str.lower() != "nan")]
    if df.empty:
        return pd.DataFrame(columns=["gene_name", "score"])
    p = df["pvalue"].astype(float).clip(lower=1e-300)
    fc = df["log2FC"].astype(float)
    if rank_stat == "log2fc":
        df["_s"] = fc
    elif rank_stat == "log2fc_neglog10p":
        df["_s"] = fc * (-np.log10(p))
    else:
        df["_s"] = np.sign(fc) * (-np.log10(p))
    df["_abs"] = df["_s"].abs()
    df = df.sort_values("_abs", ascending=False).drop_duplicates(subset=["_g"], keep="first")
    return pd.DataFrame({"gene_name": df["_g"].values, "score": df["_s"].values})


def pathway_enrichment_dotplot(
    gene_symbols: list[str],
    out_plot: str | Path,
    out_csv: str | Path | None = None,
    *,
    fdr_max: float = 0.25,
    top_n: int = 15,
    gene_sets: str = GO_BIOLOGICAL_PROCESS,
    plot_title: str | None = None,
    show: bool = False,
    dot_scale: float = 10.0,
    figsize: tuple[float, float] | None = None,
    show_ring: bool = True,
    xticklabels_rot: float = 45,
) -> bool:
    """
    Over-representation via Enrichr, then **gseapy.plot.dotplot** (color = significance, size = overlap).

    ``gene_sets`` is an Enrichr library name (e.g. ``GO_Biological_Process_2023``, ``KEGG_2021_Human``).

    Requires ``pip install gseapy`` and network access to Enrichr.
    """
    try:
        import gseapy as gp
    except ImportError:
        warnings.warn(
            "Enrichr pathway enrichment skipped: package not found. Install with:\n"
            "  pip install gseapy",
            UserWarning,
            stacklevel=2,
        )
        return False
    genes = [g for g in gene_symbols if g]
    if len(genes) < 3:
        warnings.warn(
            f"pathway enrichment skipped: need ≥3 gene symbols (got {len(genes)}).",
            UserWarning,
            stacklevel=2,
        )
        return False
    out_plot = Path(out_plot)
    out_plot.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_plot.parent / "_enrichr_cache"
    tmp.mkdir(exist_ok=True)
    try:
        enr = gp.enrichr(
            gene_list=genes,
            gene_sets=[gene_sets],
            organism="human",
            outdir=str(tmp),
            no_plot=True,
        )
    except Exception as e:
        warnings.warn(
            f"Enrichr API call failed ({gene_sets}): {e!r}. "
            "Check network / VPN; install deps: pip install gseapy requests",
            UserWarning,
            stacklevel=2,
        )
        return False
    r = enr.results
    if r is None or len(r) == 0:
        warnings.warn(
            f"Enrichr returned no rows for library {gene_sets!r}.",
            UserWarning,
            stacklevel=2,
        )
        return False
    adj_col = "Adjusted P-value"
    if adj_col not in r.columns:
        warnings.warn(
            f"Unexpected Enrichr result columns (missing {adj_col!r}).",
            UserWarning,
            stacklevel=2,
        )
        return False
    r = _parse_enrichr_overlap(r)
    if out_csv is not None:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        r.to_csv(out_csv, index=False)

    ttl = plot_title or f"Enrichr — {gene_sets}"
    h = max(5.0, 0.42 * min(top_n, max(8, len(r))))
    fs = figsize if figsize is not None else (10.0, h)
    ok_custom = _enrichr_dotplot_gene_set_sizes(
        r,
        out_plot,
        adj_col=adj_col,
        fdr_max=fdr_max,
        top_n=top_n,
        plot_title=ttl,
        show=show,
        figsize=fs,
    )
    if ok_custom:
        return True
    try:
        ax = gp.dotplot(
            r,
            column=adj_col,
            title=ttl,
            cutoff=fdr_max,
            top_term=int(top_n),
            size=float(dot_scale),
            figsize=fs,
            cmap="viridis_r",
            ofname=None,
            xticklabels_rot=xticklabels_rot,
            show_ring=show_ring,
            marker="o",
        )
        fig = ax.figure
        fig.savefig(out_plot, dpi=300, bbox_inches="tight", facecolor="white")
        if show:
            plt.show()
        plt.close(fig)
    except (ValueError, Exception) as e:
        warnings.warn(
            f"gseapy.dotplot failed for Enrichr ({gene_sets}): {e!r}. "
            f"If no terms pass adj.P ≤ {fdr_max}, try increasing ``fdr_max``.",
            UserWarning,
            stacklevel=2,
        )
        return False
    return True


def pathway_prerank_dotplot(
    res: pd.DataFrame,
    out_plot: str | Path,
    out_csv: str | Path | None = None,
    *,
    gene_sets: str = GO_BIOLOGICAL_PROCESS,
    gene_col: str = "PG.Genes",
    rank_stat: Literal["log2fc", "log2fc_neglog10p", "signed_neglog10p"] = "log2fc_neglog10p",
    fdr_cutoff: float = 0.25,
    top_term: int = 15,
    permutation_num: int = 1000,
    min_size: int = 15,
    max_size: int = 500,
    weight: float = 1.0,
    plot_title: str | None = None,
    show: bool = False,
    dot_scale: float = 10.0,
    figsize: tuple[float, float] | None = None,
    show_ring: bool = True,
    xticklabels_rot: float = 45,
    threads: int = 4,
    nes_plot: str | Path | None = None,
    nes_top_n: int = 25,
    gseaplot_top_n: int = 3,
    gseaplot_dir: str | Path | None = None,
) -> bool:
    """
    **Preranked GSEA** (``gseapy.prerank``): dotplot on ``res2d``, **NES barplot**, classical **gseaplot**
    enrichment curves for the top pathways, and CSV exports (full table + sorted-by-|NES|).

    ``weight`` follows GSEApy (1 = weighted enrichment; 0 = classic unweighted / KS-style).

    Ranking uses all proteins with valid p-value and log2FC; scores are de-duplicated by gene symbol.
    """
    try:
        import gseapy as gp
    except ImportError:
        warnings.warn(
            "Prerank pathway analysis skipped: install gseapy: pip install gseapy",
            UserWarning,
            stacklevel=2,
        )
        return False
    rnk = _de_to_prerank_dataframe(res, gene_col=gene_col, rank_stat=rank_stat)
    if len(rnk) < min_size:
        warnings.warn(
            f"Prerank skipped: need ≥{min_size} ranked genes (got {len(rnk)}).",
            UserWarning,
            stacklevel=2,
        )
        return False
    out_plot = Path(out_plot)
    out_plot.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_plot.parent / "_prerank_cache"
    tmp.mkdir(exist_ok=True)
    try:
        pre = gp.prerank(
            rnk=rnk,
            gene_sets=gene_sets,
            outdir=str(tmp / gene_sets.replace("/", "_")[:80]),
            permutation_num=permutation_num,
            min_size=min_size,
            max_size=max_size,
            weight=weight,
            no_plot=True,
            verbose=False,
            threads=threads,
        )
    except Exception as e:
        warnings.warn(
            f"gseapy.prerank failed ({gene_sets}): {e!r}",
            UserWarning,
            stacklevel=2,
        )
        return False
    df = getattr(pre, "res2d", None)
    if df is None or len(df) == 0:
        warnings.warn(f"Prerank produced no results for {gene_sets!r}.", UserWarning, stacklevel=2)
        return False
    if out_csv is not None:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        by_nes = df.reindex(df["NES"].abs().sort_values(ascending=False).index)
        by_nes_path = Path(out_csv).with_name(Path(out_csv).stem + "_sorted_by_abs_NES.csv")
        by_nes.to_csv(by_nes_path, index=False)

    col_fdr = "FDR q-val"
    if col_fdr not in df.columns:
        for alt in ("FDR q-val", "FDR", "fdr"):
            if alt in df.columns:
                col_fdr = alt
                break
        else:
            warnings.warn(
                f"Prerank results missing FDR column; columns={list(df.columns)}",
                UserWarning,
                stacklevel=2,
            )
            return False

    ttl = plot_title or f"Prerank GSEA — {gene_sets}"
    h = max(5.0, 0.42 * min(top_term, max(8, len(df))))
    fs = figsize if figsize is not None else (9.0, h)
    try:
        ax = gp.dotplot(
            df,
            column=col_fdr,
            title=ttl,
            cutoff=fdr_cutoff,
            top_term=int(top_term),
            size=float(dot_scale),
            figsize=fs,
            cmap="viridis_r",
            ofname=None,
            xticklabels_rot=xticklabels_rot,
            show_ring=show_ring,
            marker="o",
        )
        fig = ax.figure
        fig.savefig(out_plot, dpi=300, bbox_inches="tight", facecolor="white")
        if show:
            plt.show()
        plt.close(fig)
    except (ValueError, Exception) as e:
        warnings.warn(
            f"gseapy.dotplot failed for prerank ({gene_sets}): {e!r}",
            UserWarning,
            stacklevel=2,
        )

    nes_path = Path(nes_plot) if nes_plot is not None else out_plot.with_name(out_plot.stem + "_NES.png")
    _prerank_nes_barplot(
        df,
        nes_path,
        col_fdr=col_fdr,
        fdr_cutoff=fdr_cutoff,
        top_n=nes_top_n,
        title=f"{ttl} — NES",
        show=False,
    )

    gdir = Path(gseaplot_dir) if gseaplot_dir is not None else out_plot.parent / f"{out_plot.stem}_gsea_curves"
    _prerank_gseaplot_classic(pre, gdir, stem=out_plot.stem, top_n=gseaplot_top_n, show=False)

    return True


def run_de_pipeline(
    X_log2_norm: pd.DataFrame,
    proteomics_id_df: pd.DataFrame,
    sample_to_group: dict[str, str],
    *,
    plot_dir: str | Path,
    data_dir: str | Path,
    prefix: str = "DE_basic",
    top_n: int = 10,
    label_top_n: int = 20,
    show: bool = False,
) -> dict[str, Path]:
    """
    Full DE: volcano, CSVs, LFC barplot, Enrichr + **gseapy dotplot** for GO/KEGG, and **prerank GSEA + dotplot**.

    Does **not** save a table-as-image; ``top_n`` controls the LFC barplot and the exported top-DE table.
    """
    plot_dir = Path(plot_dir)
    data_dir = Path(data_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    res = differential_expression_student_t(
        X_log2_norm,
        proteomics_id_df,
        sample_to_group,
    )
    out_full = data_dir / f"{prefix}_full.csv"
    res.to_csv(out_full, index=False)

    paths: dict[str, Path] = {"de_full": out_full}
    paths["volcano"] = plot_volcano(
        res,
        plot_dir / f"{prefix}_volcano.png",
        label_top_n=label_top_n,
        show=show,
    )

    top_de = build_top_de_table(res, top_n_per_direction=top_n)
    top_de_path = data_dir / f"{prefix}_top_de.csv"
    top_de.to_csv(top_de_path, index=False)
    paths["top_de_csv"] = top_de_path

    paths["lfc_barplot"] = plot_top_lfc_barplot(
        res,
        plot_dir / f"{prefix}_lfc_barplot.png",
        top_n=top_n,
        title=f"Top {top_n} up / down (log2 FC)",
        show=show,
    )

    sub_e = res[(res["padj"] < 0.25) & (res["log2FC"] > 0)]
    genes_up: list[str] = []
    if "PG.Genes" in sub_e.columns:
        for g in sub_e["PG.Genes"].dropna():
            first = str(g).split(";")[0].strip()
            if first and first.lower() != "nan":
                genes_up.append(first)
    genes_up = list(dict.fromkeys(genes_up))

    ok_go = pathway_enrichment_dotplot(
        genes_up,
        plot_dir / f"{prefix}_pathway_GO_BP.png",
        data_dir / f"{prefix}_pathway_GO_BP.csv",
        gene_sets=GO_BIOLOGICAL_PROCESS,
        plot_title=f"GO Biological Process (elevated genes; adj.P < 0.25)",
        show=show,
    )
    if ok_go:
        paths["pathway_go_plot"] = plot_dir / f"{prefix}_pathway_GO_BP.png"
        paths["pathway_go_csv"] = data_dir / f"{prefix}_pathway_GO_BP.csv"

    ok_kegg = pathway_enrichment_dotplot(
        genes_up,
        plot_dir / f"{prefix}_pathway_KEGG.png",
        data_dir / f"{prefix}_pathway_KEGG.csv",
        gene_sets=KEGG_HUMAN,
        plot_title=f"KEGG pathways (elevated genes; adj.P < 0.25)",
        show=show,
    )
    if ok_kegg:
        paths["pathway_kegg_plot"] = plot_dir / f"{prefix}_pathway_KEGG.png"
        paths["pathway_kegg_csv"] = data_dir / f"{prefix}_pathway_KEGG.csv"

    ok_go_pre = pathway_prerank_dotplot(
        res,
        plot_dir / f"{prefix}_pathway_GO_BP_prerank.png",
        data_dir / f"{prefix}_pathway_GO_BP_prerank.csv",
        gene_sets=GO_BIOLOGICAL_PROCESS,
        plot_title="GO Biological Process — prerank GSEA (all DE genes ranked)",
        show=show,
    )
    if ok_go_pre:
        paths["pathway_go_prerank_plot"] = plot_dir / f"{prefix}_pathway_GO_BP_prerank.png"
        paths["pathway_go_prerank_csv"] = data_dir / f"{prefix}_pathway_GO_BP_prerank.csv"
        paths["pathway_go_prerank_csv_sorted"] = (
            data_dir / f"{prefix}_pathway_GO_BP_prerank_sorted_by_abs_NES.csv"
        )
        paths["pathway_go_prerank_nes_plot"] = plot_dir / f"{prefix}_pathway_GO_BP_prerank_NES.png"
        paths["pathway_go_prerank_gsea_dir"] = plot_dir / f"{prefix}_pathway_GO_BP_prerank_gsea_curves"

    ok_kegg_pre = pathway_prerank_dotplot(
        res,
        plot_dir / f"{prefix}_pathway_KEGG_prerank.png",
        data_dir / f"{prefix}_pathway_KEGG_prerank.csv",
        gene_sets=KEGG_HUMAN,
        plot_title="KEGG — prerank GSEA (all DE genes ranked)",
        show=show,
    )
    if ok_kegg_pre:
        paths["pathway_kegg_prerank_plot"] = plot_dir / f"{prefix}_pathway_KEGG_prerank.png"
        paths["pathway_kegg_prerank_csv"] = data_dir / f"{prefix}_pathway_KEGG_prerank.csv"
        paths["pathway_kegg_prerank_csv_sorted"] = (
            data_dir / f"{prefix}_pathway_KEGG_prerank_sorted_by_abs_NES.csv"
        )
        paths["pathway_kegg_prerank_nes_plot"] = plot_dir / f"{prefix}_pathway_KEGG_prerank_NES.png"
        paths["pathway_kegg_prerank_gsea_dir"] = plot_dir / f"{prefix}_pathway_KEGG_prerank_gsea_curves"

    return paths

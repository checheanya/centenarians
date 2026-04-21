"""Inflamaging marker lists: DE join, direction bars, abundance facets with stats, clustermap, top-DE panel."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Literal

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import gridspec
from scipy import stats

from .plotting import DEFAULT_CONDITION_PALETTE


def first_gene_symbol(x: Any) -> str:
    if pd.isna(x):
        return ""
    return str(x).split(";")[0].strip().upper()


def normalize_marker_list(markers: list[str], aliases: dict[str, str] | None = None) -> list[str]:
    aliases = aliases or {}
    out = [aliases.get(g.upper(), g.upper()) for g in markers]
    return list(dict.fromkeys(out))


def build_gene_maps(de_res: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    First symbol per row → unique gene lookup and mapping to matrix row labels (``de_res`` / abundance index).

    Returns
    -------
    sym_first : DataFrame indexed by gene symbol (one row per symbol).
    gene_to_matrix_row : Series mapping gene symbol → index label in ``X_log2_norm`` / ``de_res``.
    """
    _de = de_res.copy()
    _de["_sym"] = _de["PG.Genes"].map(first_gene_symbol)
    _de = _de[_de["_sym"].str.len() > 0]
    _dedup = _de.drop_duplicates("_sym", keep="first")
    sym_first = _dedup.set_index("_sym")
    gene_to_matrix_row = pd.Series(_dedup.index.to_numpy(), index=_dedup["_sym"].values)
    return sym_first, gene_to_matrix_row


def join_markers_to_de(
    sym_first: pd.DataFrame,
    markers: list[str],
) -> pd.DataFrame:
    """One row per marker with full DE columns where the protein exists."""
    rows = []
    for g in markers:
        if g in sym_first.index:
            r = sym_first.loc[g]
            if isinstance(r, pd.DataFrame):
                r = r.iloc[0]
            d = r.to_dict()
            d["marker_gene"] = g
            d["in_dataset"] = True
            rows.append(d)
        else:
            rows.append(
                {
                    "marker_gene": g,
                    "in_dataset": False,
                    "PG.Genes": np.nan,
                    "log2FC": np.nan,
                    "pvalue": np.nan,
                    "padj": np.nan,
                }
            )
    return pd.DataFrame(rows)


def _mannwhitney_p(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    try:
        return float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
    except ValueError:
        return float("nan")


def _p_stars(p: float) -> str:
    if np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _palette_rgba(palette: dict[str, str], grp_order: list[str], alpha: float) -> list[tuple[float, ...]]:
    """RGBA colors for violin bodies (more transparent than opaque box/strip)."""
    return [mcolors.to_rgba(palette[g], alpha) for g in grp_order]


def plot_inflamaging_direction_bar(
    tbl: pd.DataFrame,
    markers: list[str],
    out_path: str | Path,
    *,
    label: str,
    show: bool = False,
) -> Path:
    """Bar counts: higher / lower / not detected."""
    out_path = Path(out_path)
    sub = tbl[tbl["in_dataset"]].copy().dropna(subset=["log2FC"])
    n_list = len(markers)
    n_miss = int(tbl["in_dataset"].eq(False).sum())
    up = int((sub["log2FC"] > 0).sum())
    down = int((sub["log2FC"] < 0).sum())
    flat = int((sub["log2FC"] == 0).sum())

    fig, ax = plt.subplots(figsize=(5.5, 4))
    cats = ["Higher in\nCentenarian", "Lower in\nCentenarian", "Not detected\nin dataset"]
    vals = [up, down, n_miss]
    colors = ["#c0392b", "#2980b9", "#bdc3c7"]
    x = np.arange(len(cats))
    ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=9)
    ax.set_ylabel("Number of genes", fontsize=10)
    tot_matched = up + down + flat
    ax.set_title(
        f"Inflamaging ({label}): {n_list} list genes — "
        f"{tot_matched} in DE ({up} up, {down} down, {flat} log2FC=0), {n_miss} missing",
        fontsize=10,
    )
    ymax = max(vals) if max(vals) > 0 else 1
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02 * ymax, str(v), ha="center", va="bottom", fontsize=10)
    sns.despine(ax=ax)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def _one_gene_long(
    gene: str,
    gene_to_matrix_row: pd.Series,
    X_log2_norm: pd.DataFrame,
    abundance_cols: list[str],
    sample_to_group: dict[str, str],
) -> pd.DataFrame:
    row_idx = gene_to_matrix_row[gene]
    ser = X_log2_norm.loc[row_idx]
    rows = []
    for col in abundance_cols:
        v = ser[col]
        rows.append(
            {
                "gene": gene,
                "group": sample_to_group[col],
                "log2_abundance": float(v) if pd.notna(v) else np.nan,
            }
        )
    return pd.DataFrame(rows).dropna(subset=["log2_abundance"])


def plot_marker_abundance_facets(
    matched_genes: list[str],
    gene_to_matrix_row: pd.Series,
    X_log2_norm: pd.DataFrame,
    abundance_cols: list[str],
    sample_to_group: dict[str, str],
    de_lookup: pd.DataFrame,
    out_prefix: str | Path,
    *,
    label: str,
    palette: dict[str, str],
    grp_order: list[str] | None = None,
    chunk: int = 24,
    ncols: int = 6,
    show: bool = False,
) -> list[Path]:
    """
    Violin + box + strip per gene; titles include **Mann–Whitney p** (group difference) and **DE** log2FC / padj / p.

    ``de_lookup`` must be indexed by ``marker_gene`` (one row per gene) with columns ``log2FC``, ``pvalue``, ``padj`` where available.
    """
    grp_order = grp_order or ["Control", "Centenarian"]
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    de_lookup = de_lookup.drop_duplicates("marker_gene", keep="first")
    fc_abs = de_lookup.set_index("marker_gene")["log2FC"].abs()
    genes_order = sorted(matched_genes, key=lambda x: fc_abs.get(x, 0), reverse=True)

    saved: list[Path] = []
    for part, start in enumerate(range(0, len(genes_order), chunk)):
        chunk_genes = genes_order[start : start + chunk]
        long_parts = []
        for g in chunk_genes:
            long_parts.append(
                _one_gene_long(g, gene_to_matrix_row, X_log2_norm, abundance_cols, sample_to_group)
            )
        long_df = pd.concat(long_parts, ignore_index=True)
        if long_df.empty:
            continue

        dfp = long_df[long_df["gene"].isin(chunk_genes)]
        n = len(chunk_genes)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(2.5 * ncols, 2.85 * nrows), squeeze=False)

        for idx, gene in enumerate(chunk_genes):
            rr, cc = divmod(idx, ncols)
            ax = axes[rr, cc]
            dg = dfp[dfp["gene"] == gene]
            v0 = dg.loc[dg["group"] == grp_order[0], "log2_abundance"].values
            v1 = dg.loc[dg["group"] == grp_order[1], "log2_abundance"].values
            mw_p = _mannwhitney_p(v0, v1)

            sns.violinplot(
                data=dg,
                x="group",
                y="log2_abundance",
                order=grp_order,
                palette=_palette_rgba(palette, grp_order, 0.38),
                ax=ax,
                cut=0,
                inner=None,
                linewidth=0.6,
                saturation=1.0,
            )
            sns.boxplot(
                data=dg,
                x="group",
                y="log2_abundance",
                order=grp_order,
                ax=ax,
                width=0.22,
                showcaps=True,
                showfliers=False,
                boxprops=dict(facecolor="white", edgecolor="0.35", alpha=1.0),
                medianprops=dict(color="black", linewidth=1.0),
                whiskerprops=dict(color="0.35"),
            )
            sns.stripplot(
                data=dg,
                x="group",
                y="log2_abundance",
                order=grp_order,
                hue="group",
                hue_order=grp_order,
                palette=palette,
                dodge=False,
                jitter=0.12,
                alpha=0.72,
                size=2,
                ax=ax,
                legend=False,
                linewidth=0,
            )
            ax.set_xlabel("")
            ax.set_ylabel("log2 abund." if cc == 0 else "", fontsize=7)

            de_row = de_lookup[de_lookup["marker_gene"] == gene]
            if len(de_row):
                de_row = de_row.iloc[0]
                lfc = de_row.get("log2FC", np.nan)
                pv = de_row.get("pvalue", np.nan)
                q = de_row.get("padj", np.nan)
                stat_txt = (
                    f"MW p={mw_p:.2e} {_p_stars(mw_p)}\n"
                    f"DE: log2FC={lfc:.3f}, p={pv:.2e}, padj={q:.2e}"
                )
            else:
                stat_txt = f"MW p={mw_p:.2e} {_p_stars(mw_p)}"

            ylim = ax.get_ylim()
            ax.text(
                0.02,
                0.98,
                stat_txt,
                transform=ax.transAxes,
                fontsize=5.5,
                va="top",
                ha="left",
                linespacing=1.15,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.88, edgecolor="0.75"),
            )
            ax.set_ylim(ylim)

            ax.set_title(gene, fontsize=8)

        for j in range(len(chunk_genes), nrows * ncols):
            rr, cc = divmod(j, ncols)
            axes[rr, cc].set_visible(False)

        fig.suptitle(
            f"Inflamaging ({label}): log2 abundance (median-centered) + Mann–Whitney (groups) & DE stats",
            fontsize=11,
            y=1.01,
        )
        plt.tight_layout()
        suf = f"_{part + 1}" if len(genes_order) > chunk else ""
        pth = out_prefix.parent / f"{out_prefix.name}{suf}.png"
        fig.savefig(pth, dpi=300, bbox_inches="tight", facecolor="white")
        if show:
            plt.show()
        plt.close(fig)
        saved.append(pth)
    return saved


def plot_inflamaging_clustermap(
    genes: list[str],
    gene_to_matrix_row: pd.Series,
    X_log2_norm: pd.DataFrame,
    abundance_cols: list[str],
    sample_to_group: dict[str, str],
    out_path: str | Path,
    *,
    title: str = "Inflamaging markers",
    zscore_rows: bool = True,
    group_palette: dict[str, str] | None = None,
    figsize: tuple[float, float] = (14.0, 11.0),
    show: bool = False,
) -> Path | None:
    """
    Clustered heatmap: genes × samples (log2 abundances). Column color bar = group.
    Rows are z-scored per gene if ``zscore_rows``.
    """
    out_path = Path(out_path)
    genes = [g for g in genes if g in gene_to_matrix_row.index]
    if len(genes) < 2:
        warnings.warn("Clustermap needs ≥2 genes present in data.", UserWarning, stacklevel=2)
        return None

    mat_rows = []
    for g in genes:
        rid = gene_to_matrix_row[g]
        mat_rows.append(X_log2_norm.loc[rid, abundance_cols].astype(float).values)
    mat = np.vstack(mat_rows)
    if zscore_rows:
        mat = stats.zscore(mat, axis=1, nan_policy="omit")
        mat = np.nan_to_num(mat, nan=0.0)

    df = pd.DataFrame(mat, index=genes, columns=abundance_cols)

    groups = [sample_to_group[c] for c in abundance_cols]
    ug = sorted(set(groups))
    group_palette = group_palette or dict(DEFAULT_CONDITION_PALETTE)
    lut = {g: group_palette.get(g, "#888888") for g in ug}
    col_colors = pd.Series(groups, index=abundance_cols).map(lut)

    g = sns.clustermap(
        df,
        col_colors=col_colors,
        cmap="vlag",
        center=0 if zscore_rows else None,
        figsize=figsize,
        xticklabels=False,
        yticklabels=True,
        dendrogram_ratio=(0.12, 0.12),
        cbar_pos=(0.02, 0.85, 0.02, 0.12),
        linewidths=0,
    )
    g.ax_heatmap.set_ylabel("Gene (row z-score)" if zscore_rows else "Gene")
    g.ax_heatmap.set_xlabel("Samples (clustered)")
    g.fig.suptitle(title, y=1.02, fontsize=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    g.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    plt.close("all")
    return out_path


def rank_inflamaging_de_genes(
    tbl: pd.DataFrame,
    *,
    rank_by: Literal["abs_log2fc", "padj"] = "abs_log2fc",
    top_n: int = 15,
) -> pd.DataFrame:
    """Return top DE hits among markers (``in_dataset`` with valid log2FC)."""
    sub = tbl[tbl["in_dataset"]].copy().dropna(subset=["log2FC"])
    if sub.empty:
        return sub
    if rank_by == "padj":
        sub = sub.dropna(subset=["padj"]).sort_values("padj", ascending=True, na_position="last")
    else:
        sub = sub.assign(_abs=np.abs(sub["log2FC"].astype(float)))
        sub = sub.sort_values("_abs", ascending=False).drop(columns=["_abs"])
    return sub.head(int(top_n)).reset_index(drop=True)


def print_top_inflamaging_de(tbl: pd.DataFrame, *, label: str, top_n: int = 15) -> pd.DataFrame:
    """Print and return ranked table."""
    top = rank_inflamaging_de_genes(tbl, top_n=top_n)
    print(f"\n=== Top {len(top)} inflamaging DE genes ({label}) by |log2FC| ===\n")
    cols = [c for c in ["marker_gene", "log2FC", "pvalue", "padj", "PG.Genes"] if c in top.columns]
    print(top[cols].to_string(index=False))
    return top


def plot_top_inflamaging_de_panel(
    top_df: pd.DataFrame,
    gene_to_matrix_row: pd.Series,
    X_log2_norm: pd.DataFrame,
    abundance_cols: list[str],
    sample_to_group: dict[str, str],
    out_path: str | Path,
    *,
    palette: dict[str, str],
    grp_order: list[str] | None = None,
    ncols: int = 4,
    show: bool = False,
) -> Path:
    """
    Single figure: one column of violin+box+strip per top gene (grid), with stats in each panel.
    """
    grp_order = grp_order or ["Control", "Centenarian"]
    genes = top_df["marker_gene"].tolist()
    n = len(genes)
    if n == 0:
        raise ValueError("top_df has no rows")

    nrows = int(np.ceil(n / ncols))
    fig_w = 2.8 * ncols
    fig_h = 3.0 * nrows
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = gridspec.GridSpec(nrows, ncols, figure=fig, hspace=0.45, wspace=0.35)

    de_lookup = top_df

    for i, gene in enumerate(genes):
        r, c = divmod(i, ncols)
        ax = fig.add_subplot(gs[r, c])
        dg = _one_gene_long(gene, gene_to_matrix_row, X_log2_norm, abundance_cols, sample_to_group)
        v0 = dg.loc[dg["group"] == grp_order[0], "log2_abundance"].values
        v1 = dg.loc[dg["group"] == grp_order[1], "log2_abundance"].values
        mw_p = _mannwhitney_p(v0, v1)

        sns.violinplot(
            data=dg,
            x="group",
            y="log2_abundance",
            order=grp_order,
            palette=_palette_rgba(palette, grp_order, 0.38),
            ax=ax,
            cut=0,
            inner=None,
            linewidth=0.55,
            saturation=1.0,
        )
        sns.boxplot(
            data=dg,
            x="group",
            y="log2_abundance",
            order=grp_order,
            ax=ax,
            width=0.2,
            showcaps=True,
            showfliers=False,
            boxprops=dict(facecolor="white", edgecolor="0.35", alpha=1.0),
            medianprops=dict(color="black", linewidth=0.9),
        )
        sns.stripplot(
            data=dg,
            x="group",
            y="log2_abundance",
            order=grp_order,
            hue="group",
            hue_order=grp_order,
            palette=palette,
            dodge=False,
            jitter=0.12,
            alpha=0.72,
            size=2.2,
            ax=ax,
            legend=False,
            linewidth=0,
        )
        de_row = de_lookup[de_lookup["marker_gene"] == gene].iloc[0]
        stat_txt = (
            f"MW p={mw_p:.2e} {_p_stars(mw_p)}\n"
            f"log2FC={de_row['log2FC']:.3f}  padj={de_row['padj']:.2e}"
        )
        ax.text(
            0.03,
            0.97,
            stat_txt,
            transform=ax.transAxes,
            fontsize=6,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.92, edgecolor="0.7"),
        )
        ax.set_title(gene, fontweight="semibold", fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("log2 abund." if c == 0 else "", fontsize=8)

    fig.suptitle("Top inflamaging markers by |log2FC| (Centenarian vs Control)", fontsize=12, y=1.02)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    plt.close(fig)
    return out_path

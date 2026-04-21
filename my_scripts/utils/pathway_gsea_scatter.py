"""
KEGG prerank GSEA: merge two ranked lists (e.g. UKBB aging vs cohort DE) and plot NES scatter (Figure 3 panel e style).

**Ranking (matches manuscript):** genes are ranked by the raw effect only — UKBB uses **Age_Beta**;
centenarian cohort uses **log2FC** (or Swiss ``log2FoldChange``, NECS ``log2(FC.cont2cent)``). We do **not**
multiply by −log10(p); that over-weights strong p-values and inflates |NES| so the cloud no longer sits near 0.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats

from .de_analysis import KEGG_HUMAN, _gene_symbol_series

# Match make_figure3_v3.py panel e
C_CARDIAC = "#E64B35"
C_IMMUNE = "#4DBBD5"
C_METABOLISM = "#00A087"
C_OTHER = "#B0B0B0"
C_ESCAPE = "#F39B7F"

# ── Panel e: same pathway sets as make_figure3_v3.py (GMT-style KEGG_* ids) ──
FIGURE3_CARDIAC_PATHWAYS = frozenset(
    {
        "KEGG_HYPERTROPHIC_CARDIOMYOPATHY_HCM",
        "KEGG_DILATED_CARDIOMYOPATHY",
        "KEGG_ARRHYTHMOGENIC_RIGHT_VENTRICULAR_CARDIOMYOPATHY_ARVC",
        "KEGG_VIRAL_MYOCARDITIS",
        "KEGG_VASCULAR_SMOOTH_MUSCLE_CONTRACTION",
    }
)
FIGURE3_IMMUNE_PATHWAYS = frozenset(
    {
        "KEGG_INTESTINAL_IMMUNE_NETWORK_FOR_IGA_PRODUCTION",
        "KEGG_CYTOKINE_CYTOKINE_RECEPTOR_INTERACTION",
        "KEGG_COMPLEMENT_AND_COAGULATION_CASCADES",
        "KEGG_TOLL_LIKE_RECEPTOR_SIGNALING_PATHWAY",
        "KEGG_B_CELL_RECEPTOR_SIGNALING_PATHWAY",
        "KEGG_T_CELL_RECEPTOR_SIGNALING_PATHWAY",
        "KEGG_NATURAL_KILLER_CELL_MEDIATED_CYTOTOXICITY",
        "KEGG_FC_GAMMA_R_MEDIATED_PHAGOCYTOSIS",
        "KEGG_FC_EPSILON_RI_SIGNALING_PATHWAY",
        "KEGG_LEUKOCYTE_TRANSENDOTHELIAL_MIGRATION",
        "KEGG_CHEMOKINE_SIGNALING_PATHWAY",
        "KEGG_JAK_STAT_SIGNALING_PATHWAY",
        "KEGG_HEMATOPOIETIC_CELL_LINEAGE",
        "KEGG_ANTIGEN_PROCESSING_AND_PRESENTATION",
        "KEGG_NOD_LIKE_RECEPTOR_SIGNALING_PATHWAY",
        "KEGG_RIG_I_LIKE_RECEPTOR_SIGNALING_PATHWAY",
        "KEGG_SYSTEMIC_LUPUS_ERYTHEMATOSUS",
        "KEGG_LEISHMANIA_INFECTION",
    }
)
FIGURE3_METABOLISM_PATHWAYS = frozenset(
    {
        "KEGG_PPAR_SIGNALING_PATHWAY",
        "KEGG_ADIPOCYTOKINE_SIGNALING_PATHWAY",
        "KEGG_INSULIN_SIGNALING_PATHWAY",
        "KEGG_GLYCEROPHOSPHOLIPID_METABOLISM",
        "KEGG_GLUTATHIONE_METABOLISM",
        "KEGG_PURINE_METABOLISM",
        "KEGG_PYRIMIDINE_METABOLISM",
        "KEGG_PEROXISOME",
        "KEGG_LYSOSOME",
    }
)
FIGURE3_ESCAPE_PATHWAYS = frozenset(
    {
        "KEGG_HYPERTROPHIC_CARDIOMYOPATHY_HCM",
        "KEGG_DILATED_CARDIOMYOPATHY",
        "KEGG_INTESTINAL_IMMUNE_NETWORK_FOR_IGA_PRODUCTION",
    }
)

# Enrichr "Term" → figure3 KEGG id when automatic slug ≠ GMT id
FIGURE3_PATHWAY_ID_ALIASES: dict[str, str] = {
    "KEGG_LEISHMANIASIS": "KEGG_LEISHMANIA_INFECTION",
    "KEGG_HYPERTROPHIC_CARDIOMYOPATHY": "KEGG_HYPERTROPHIC_CARDIOMYOPATHY_HCM",
    "KEGG_ARRHYTHMOGENIC_RIGHT_VENTRICULAR_CARDIOMYOPATHY": "KEGG_ARRHYTHMOGENIC_RIGHT_VENTRICULAR_CARDIOMYOPATHY_ARVC",
}

# Escape / concordant text labels (make_figure3_v3 panel e)
FIGURE3_PW_LABEL_MAP: dict[str, str] = {
    "KEGG_HYPERTROPHIC_CARDIOMYOPATHY_HCM": "HCM",
    "KEGG_DILATED_CARDIOMYOPATHY": "DCM",
    "KEGG_INTESTINAL_IMMUNE_NETWORK_FOR_IGA_PRODUCTION": "IgA production",
}
FIGURE3_CONCORDANT_LABEL_MAP: dict[str, str] = {
    "KEGG_COMPLEMENT_AND_COAGULATION_CASCADES": "Complement &\ncoagulation",
    "KEGG_CYTOKINE_CYTOKINE_RECEPTOR_INTERACTION": "Cytokine-cytokine\ninteraction",
    "KEGG_TGF_BETA_SIGNALING_PATHWAY": r"TGF-$\beta$",
    "KEGG_ERBB_SIGNALING_PATHWAY": "ErbB",
    "KEGG_MAPK_SIGNALING_PATHWAY": "MAPK",
    "KEGG_ARRHYTHMOGENIC_RIGHT_VENTRICULAR_CARDIOMYOPATHY_ARVC": "ARVC",
    "KEGG_ECM_RECEPTOR_INTERACTION": "ECM-receptor",
    "KEGG_APOPTOSIS": "Apoptosis",
}


def figure3_pathway_id(term: str) -> str:
    """
    Map a ``Pathway`` value to the GMT-style id used in make_figure3_v3 (``KEGG_*``).
    Accepts either native ``KEGG_...`` ids (sanju CSV) or Enrichr/gseapy human-readable terms.
    """
    s = str(term).strip()
    u = s.upper()
    if u.startswith("KEGG_"):
        pid = u
    else:
        s2 = re.sub(r"\s*\([^)]*\)\s*$", "", s).strip()
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", s2).strip("_").upper()
        pid = f"KEGG_{slug}" if slug else u
    return FIGURE3_PATHWAY_ID_ALIASES.get(pid, pid)


def is_highlight_escape_term(term: str) -> bool:
    """True for the three escape pathways highlighted as diamonds in make_figure3_v3 panel e."""
    return figure3_pathway_id(term) in FIGURE3_ESCAPE_PATHWAYS


def get_pw_color_figure3(term: str) -> str:
    """Exact logic order as make_figure3_v3 ``get_pw_color`` (cardiac → immune → metabolism → other)."""
    pid = figure3_pathway_id(term)
    if pid in FIGURE3_CARDIAC_PATHWAYS:
        return C_CARDIAC
    if pid in FIGURE3_IMMUNE_PATHWAYS:
        return C_IMMUNE
    if pid in FIGURE3_METABOLISM_PATHWAYS:
        return C_METABOLISM
    return C_OTHER


def get_pw_color(term: str) -> str:
    """Alias for ``get_pw_color_figure3`` (panel e)."""
    return get_pw_color_figure3(term)


def build_prerank_from_effect_p(
    df: pd.DataFrame,
    *,
    gene_col: str,
    effect_col: str,
    p_col: str,
    clip_p: float = 1e-300,
) -> pd.DataFrame:
    d = df.dropna(subset=[gene_col, effect_col, p_col]).copy()
    p = pd.to_numeric(d[p_col], errors="coerce").clip(lower=clip_p)
    eff = pd.to_numeric(d[effect_col], errors="coerce")
    d["_s"] = eff * (-np.log10(p))
    d["_g"] = d[gene_col].astype(str).str.strip()
    d = d[(d["_g"].str.len() > 0) & (d["_g"].str.lower() != "nan")]
    d["_abs"] = d["_s"].abs()
    d = d.sort_values("_abs", ascending=False).drop_duplicates("_g", keep="first")
    return pd.DataFrame({"gene_name": d["_g"].values, "score": d["_s"].values})


def build_prerank_effect_only(
    df: pd.DataFrame,
    *,
    gene_col: str,
    effect_col: str,
) -> pd.DataFrame:
    """
    Prerank table for GSEA: **score = raw effect only** (e.g. Age_Beta or log2FC).
    One row per gene (largest |effect| wins). Missing effects dropped.
    """
    d = df.dropna(subset=[gene_col, effect_col]).copy()
    eff = pd.to_numeric(d[effect_col], errors="coerce")
    d = d.assign(_s=eff)
    d = d[np.isfinite(d["_s"])]
    d["_g"] = d[gene_col].astype(str).str.strip()
    d = d[(d["_g"].str.len() > 0) & (d["_g"].str.lower() != "nan")]
    d["_abs"] = d["_s"].abs()
    d = d.sort_values("_abs", ascending=False).drop_duplicates("_g", keep="first")
    return pd.DataFrame({"gene_name": d["_g"].values, "score": d["_s"].values})


def build_ukbb_age_prerank(
    ukbb_preprocessed: pd.DataFrame,
    *,
    gene_col: str = "gene_symbol_match",
    beta_col: str = "Age_Beta",
) -> pd.DataFrame:
    """UKBB aging GSEA: rank genes by **Age_Beta** only."""
    return build_prerank_effect_only(
        ukbb_preprocessed,
        gene_col=gene_col,
        effect_col=beta_col,
    )


def build_ciao_de_prerank(de_full: pd.DataFrame, *, gene_col: str = "PG.Genes") -> pd.DataFrame:
    """CIAO centenarian GSEA: rank genes by **log2FC** only."""
    d = de_full.copy()
    g = _gene_symbol_series(d, gene_col=gene_col)
    d["_gene"] = g.astype(str).str.strip()
    d = d[d["_gene"].str.len() > 0]
    return build_prerank_effect_only(d, gene_col="_gene", effect_col="log2FC")


def build_swiss_prerank(swiss_df: pd.DataFrame) -> pd.DataFrame:
    """Swiss100 GSEA: rank genes by **log2FoldChange** only."""
    return build_prerank_effect_only(swiss_df, gene_col="Gene", effect_col="log2FoldChange")


def build_necs_prerank(necs_df: pd.DataFrame) -> pd.DataFrame:
    """NECS GSEA: rank genes by **log2(FC.cont2cent)** only."""
    d = necs_df.copy()
    d["Gene"] = d["geneID"].astype(str).str.strip()
    fc = pd.to_numeric(d["FC.cont2cent"], errors="coerce").clip(lower=1e-9)
    d["_log2fc"] = np.log2(fc)
    return build_prerank_effect_only(d, gene_col="Gene", effect_col="_log2fc")


def run_kegg_prerank(
    rnk: pd.DataFrame,
    out_csv: str | Path | None,
    *,
    gene_sets: str = KEGG_HUMAN,
    permutation_num: int = 1000,
    min_size: int = 15,
    max_size: int = 500,
    weight: float = 1.0,
    threads: int = 4,
    seed: int = 42,
) -> pd.DataFrame:
    """Run gseapy.prerank; return res2d (and optionally save CSV)."""
    import gseapy as gp

    if len(rnk) < min_size:
        raise ValueError(f"Need ≥{min_size} ranked genes, got {len(rnk)}")
    out_csv = Path(out_csv) if out_csv is not None else None
    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp = (out_csv.parent if out_csv else Path.cwd()) / "_prerank_cache_gsea_pair"
    tmp.mkdir(exist_ok=True)
    pre = gp.prerank(
        rnk=rnk,
        gene_sets=gene_sets,
        outdir=str(tmp / re.sub(r"[^\w\-]+", "_", gene_sets)[:60]),
        permutation_num=permutation_num,
        min_size=min_size,
        max_size=max_size,
        weight=weight,
        no_plot=True,
        verbose=False,
        threads=threads,
        seed=seed,
    )
    df = getattr(pre, "res2d", None)
    if df is None or len(df) == 0:
        raise RuntimeError("gseapy.prerank returned empty res2d")
    df = df.copy()
    if out_csv is not None:
        df.to_csv(out_csv, index=False)
    return df


def _nom_p_col(df: pd.DataFrame) -> str:
    for c in ("NOM p-val", "NOM p-value", "P-value", "pval", "Pval"):
        if c in df.columns:
            return c
    raise KeyError(f"No nominal p column in {list(df.columns)}")


def _fdr_col(df: pd.DataFrame) -> str | None:
    for c in ("FDR q-val", "FDR q-val ", "FDRqval", "fdr_qval"):
        if c in df.columns:
            return c
    return None


def merge_two_prerank_tables(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    *,
    label_left: str = "UKBB",
    label_right: str = "Cohort",
) -> pd.DataFrame:
    """Inner join on KEGG term; columns compatible with Figure 3e (NES_*, pval_*, fdr_*)."""
    term_col = "Term" if "Term" in df_left.columns else df_left.columns[0]
    if term_col not in df_right.columns:
        raise KeyError(f"Expected {term_col!r} in both tables")
    pc_l = _nom_p_col(df_left)
    pc_r = _nom_p_col(df_right)
    fdr_l = _fdr_col(df_left)
    fdr_r = _fdr_col(df_right)
    cols_l = [term_col, "NES", pc_l]
    if fdr_l:
        cols_l.append(fdr_l)
    a = df_left[cols_l].rename(
        columns={term_col: "Pathway", "NES": "NES_Nature", pc_l: "pval_Nature"}
    )
    if fdr_l:
        a = a.rename(columns={fdr_l: "fdr_Nature"})
    else:
        a["fdr_Nature"] = np.nan

    cols_r = [term_col, "NES", pc_r]
    if fdr_r:
        cols_r.append(fdr_r)
    b = df_right[cols_r].rename(
        columns={term_col: "Pathway", "NES": "NES_Cent", pc_r: "pval_Cent"}
    )
    if fdr_r:
        b = b.rename(columns={fdr_r: "fdr_Cent"})
    else:
        b["fdr_Cent"] = np.nan

    m = a.merge(b, on="Pathway", how="inner")
    m.attrs["label_left"] = label_left
    m.attrs["label_right"] = label_right
    return m


def plot_nes_scatter_figure3e_style(
    gsea: pd.DataFrame,
    out_path: str | Path,
    *,
    xlabel: str = "NES (UKB aging)",
    ylabel: str = "NES (cohort)",
    title: str = "Pathway-level GSEA: aging vs centenarian",
    figsize: tuple[float, float] = (5.2, 4.8),
) -> Path:
    """Scatter NES_Nature vs NES_Cent; same pathway lists, escape set, sizes, and labels as make_figure3_v3 panel e."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    g = gsea.copy()
    g["NES_Nature"] = pd.to_numeric(g["NES_Nature"], errors="coerce")
    g["NES_Cent"] = pd.to_numeric(g["NES_Cent"], errors="coerce")
    g["pval_Nature"] = pd.to_numeric(g["pval_Nature"], errors="coerce")
    g["pval_Cent"] = pd.to_numeric(g["pval_Cent"], errors="coerce")
    g["_fig3_pid"] = g["Pathway"].map(figure3_pathway_id)
    g["pw_color"] = g["Pathway"].map(get_pw_color_figure3)
    g["is_escape"] = g["_fig3_pid"].isin(FIGURE3_ESCAPE_PATHWAYS)

    g["min_pval"] = g[["pval_Nature", "pval_Cent"]].min(axis=1).clip(lower=1e-10)
    g["neg_log10p"] = -np.log10(g["min_pval"])
    nlmax = float(g["neg_log10p"].max()) if len(g) else 1.0
    if not np.isfinite(nlmax) or nlmax <= 0:
        nlmax = 1.0

    fig, ax = plt.subplots(figsize=figsize)
    non_esc = g[~g["is_escape"]]
    esc_pw = g[g["is_escape"]]

    for _, row in non_esc.iterrows():
        sz = max(15, 60 * (row["neg_log10p"] / nlmax))
        ax.scatter(
            row["NES_Nature"],
            row["NES_Cent"],
            c=row["pw_color"],
            s=sz,
            alpha=0.55,
            edgecolors="white",
            linewidths=0.3,
            zorder=2,
        )

    for _, row in esc_pw.iterrows():
        sz = max(40, 100 * (row["neg_log10p"] / nlmax))
        ax.scatter(
            row["NES_Nature"],
            row["NES_Cent"],
            c=row["pw_color"],
            s=sz,
            alpha=0.95,
            edgecolors="black",
            linewidths=0.8,
            zorder=6,
            marker="D",
        )

    ax.axhline(0, color="#888888", lw=0.4, ls="--")
    ax.axvline(0, color="#888888", lw=0.4, ls="--")
    lim_e = max(abs(g["NES_Nature"]).max(), abs(g["NES_Cent"]).max()) * 1.15
    ax.plot([-lim_e, lim_e], [-lim_e, lim_e], color="grey", lw=0.5, ls=":", alpha=0.5)
    ax.fill_between([0, lim_e], -lim_e, 0, color=C_ESCAPE, alpha=0.04, zorder=0)

    texts = []
    for _, row in esc_pw.iterrows():
        label = FIGURE3_PW_LABEL_MAP.get(row["_fig3_pid"], "")
        if label:
            t = ax.text(
                row["NES_Nature"],
                row["NES_Cent"],
                label,
                fontsize=5.5,
                fontweight="bold",
                color="#333333",
                ha="center",
                va="center",
            )
            texts.append(t)

    for pw_id, label in FIGURE3_CONCORDANT_LABEL_MAP.items():
        hit = g[g["_fig3_pid"] == pw_id]
        if len(hit):
            row = hit.iloc[0]
            t = ax.text(
                row["NES_Nature"],
                row["NES_Cent"],
                label,
                fontsize=4.5,
                color="#555555",
                ha="center",
                va="center",
            )
            texts.append(t)

    if texts:
        try:
            from adjustText import adjust_text

            adjust_text(
                texts,
                ax=ax,
                arrowprops=dict(arrowstyle="-", color="#555555", lw=0.3),
                force_text=(0.6, 0.6),
                force_points=(0.3, 0.3),
                expand_text=(1.3, 1.5),
                expand_points=(1.2, 1.2),
                lim=500,
            )
        except Exception:
            pass

    rho_e, pval_e = stats.spearmanr(g["NES_Nature"], g["NES_Cent"])
    pstr = f"p = {pval_e:.1e}" if pval_e < 0.001 else f"p = {pval_e:.3f}"
    ax.text(
        0.03,
        0.97,
        f"$\\rho$ = {rho_e:.2f}\n{pstr}\nn = {len(g)} pathways",
        transform=ax.transAxes,
        fontsize=5,
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="#999999", lw=0.3),
    )

    ax.set_xlabel(xlabel, fontsize=7)
    ax.set_ylabel(ylabel, fontsize=7)
    ax.set_title(title, fontsize=7, fontweight="bold")

    leg_e = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_CARDIAC, markersize=4, label="Cardiac"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_IMMUNE, markersize=4, label="Immune"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_METABOLISM, markersize=4, label="Metabolism"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_OTHER, markersize=4, label="Other"),
        Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            markerfacecolor="none",
            markeredgecolor="black",
            markersize=5,
            markeredgewidth=0.8,
            label="Escape pathway",
        ),
    ]
    ax.legend(
        handles=leg_e,
        loc="lower right",
        fontsize=4.5,
        framealpha=0.85,
        edgecolor="#999999",
        borderpad=0.3,
        handletextpad=0.3,
        labelspacing=0.3,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def run_pair_and_plot(
    rnk_left: pd.DataFrame,
    rnk_right: pd.DataFrame,
    *,
    out_merged_csv: Path,
    out_plot: Path,
    cache_left_csv: Path,
    cache_right_csv: Path,
    label_left: str,
    label_right: str,
    xlabel: str,
    ylabel: str,
    title: str,
    reuse_left: bool = False,
    reuse_right: bool = False,
) -> tuple[pd.DataFrame, Path]:
    if reuse_left and cache_left_csv.is_file():
        df_l = pd.read_csv(cache_left_csv)
    else:
        df_l = run_kegg_prerank(rnk_left, cache_left_csv)
    if reuse_right and cache_right_csv.is_file():
        df_r = pd.read_csv(cache_right_csv)
    else:
        df_r = run_kegg_prerank(rnk_right, cache_right_csv)

    merged = merge_two_prerank_tables(df_l, df_r, label_left=label_left, label_right=label_right)
    out_merged_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_merged_csv, index=False)
    plot_nes_scatter_figure3e_style(
        merged,
        out_plot,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
    )
    return merged, out_plot

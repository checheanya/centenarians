"""PCA scatter plots for proteomics (color by condition or age)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .plotting import DEFAULT_CONDITION_PALETTE


def map_prot_columns_to_meta(
    meta_df: pd.DataFrame,
    abundance_cols: list[str],
    *,
    prot_col: str = "Prot_Col",
    condition_col: str = "Condition",
    age_col: str = "Age",
) -> tuple[pd.Series, pd.Series]:
    """
    Align metadata to proteomics sample (column) names.

    Returns
    -------
    condition : Series indexed by abundance column name
    age : Series indexed by abundance column name (numeric, may contain NaN)
    """
    m = meta_df[[prot_col, condition_col, age_col]].drop_duplicates(subset=[prot_col])
    m = m.set_index(prot_col)
    cond_list: list[Any] = []
    age_list: list[float] = []
    for c in abundance_cols:
        if c not in m.index:
            cond_list.append(np.nan)
            age_list.append(np.nan)
        else:
            cond_list.append(m.loc[c, condition_col])
            age_list.append(pd.to_numeric(m.loc[c, age_col], errors="coerce"))
    return (
        pd.Series(cond_list, index=abundance_cols, name=condition_col),
        pd.Series(age_list, index=abundance_cols, name=age_col, dtype=float),
    )


def plot_pca_by_condition(
    pca_df: pd.DataFrame,
    *,
    out_path: str | Path,
    palette: dict[str, str],
    group_col: str = "group",
    x: str = "PC1",
    y: str = "PC2",
    title: str = "PCA — color by condition",
    figsize: tuple[float, float] = (7, 6),
    explained_variance_ratio: np.ndarray | None = None,
    show: bool = False,
) -> Path:
    """Scatter PC1 vs PC2 with discrete condition colors."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=figsize)
    sub = pca_df.dropna(subset=[group_col])
    sns.scatterplot(
        data=sub,
        x=x,
        y=y,
        hue=group_col,
        palette={k: palette.get(k, "#888888") for k in sub[group_col].dropna().unique()},
        hue_order=sorted(sub[group_col].dropna().unique(), key=str),
        s=55,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
        ax=ax,
    )
    ev = explained_variance_ratio
    if ev is not None and len(ev) >= 2:
        ax.set_xlabel(f"{x} ({ev[0]*100:.1f}% var.)")
        ax.set_ylabel(f"{y} ({ev[1]*100:.1f}% var.)")
    else:
        ax.set_xlabel(x)
        ax.set_ylabel(y)
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, title=group_col)
    sns.despine(ax=ax)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def plot_pca_by_age(
    pca_df: pd.DataFrame,
    *,
    age_by_sample: pd.Series,
    out_path: str | Path,
    sample_col: str = "sample",
    x: str = "PC1",
    y: str = "PC2",
    title: str = "PCA — color by age",
    figsize: tuple[float, float] = (7.2, 6),
    cmap: str = "viridis",
    explained_variance_ratio: np.ndarray | None = None,
    show: bool = False,
) -> Path:
    """Scatter PC1 vs PC2 with continuous age coloring."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot_df = pca_df[[sample_col, x, y]].merge(
        age_by_sample.rename("age"),
        left_on=sample_col,
        right_index=True,
        how="left",
    )
    plot_df = plot_df.dropna(subset=["age"])
    fig, ax = plt.subplots(figsize=figsize)
    sc = ax.scatter(
        plot_df[x],
        plot_df[y],
        c=plot_df["age"],
        cmap=cmap,
        s=55,
        alpha=0.85,
        edgecolors="white",
        linewidths=0.4,
    )
    plt.colorbar(sc, ax=ax, label="Age", shrink=0.82)
    ev = explained_variance_ratio
    if ev is not None and len(ev) >= 2:
        ax.set_xlabel(f"{x} ({ev[0]*100:.1f}% var.)")
        ax.set_ylabel(f"{y} ({ev[1]*100:.1f}% var.)")
    else:
        ax.set_xlabel(x)
        ax.set_ylabel(y)
    ax.set_title(title)
    sns.despine(ax=ax)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def run_pca_proteomics_exports(
    proteomics_filtered: pd.DataFrame,
    abundance_cols: list[str],
    meta_full: pd.DataFrame,
    *,
    out_dir: str | Path,
    palette: dict[str, str] | None = None,
    prot_col: str = "Prot_Col",
    condition_col: str = "Condition",
    age_col: str = "Age",
    n_components: int = 2,
    random_state: int = 42,
    show: bool = False,
) -> dict[str, Path]:
    """
    PCA on log2 median-centered matrix (samples = columns), save condition + age figures.

    ``proteomics_filtered`` must contain at least ``abundance_cols`` (sample-level columns).
    """
    from .preprocessing import log2_abundance, median_center_samples, sample_pca

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    X = proteomics_filtered[abundance_cols].copy()
    X_log2 = log2_abundance(X, zero_as_nan=True)
    X_log2_norm, _ = median_center_samples(X_log2)

    pca_df, _pca_model, evr = sample_pca(
        X_log2_norm,
        n_components=n_components,
        random_state=random_state,
    )

    cond_ser, age_ser = map_prot_columns_to_meta(
        meta_full,
        abundance_cols,
        prot_col=prot_col,
        condition_col=condition_col,
        age_col=age_col,
    )
    samp = pca_df["sample"].astype(str)
    grp = samp.map(lambda s: cond_ser.get(s, np.nan))
    fallback = samp.map(
        lambda s: "Control" if "CTRL" in str(s).upper() else "Centenarian"
    )
    pca_df["group"] = grp.where(grp.notna(), fallback).astype(str).str.strip()

    pal = palette or dict(DEFAULT_CONDITION_PALETTE)
    paths: dict[str, Path] = {}
    paths["pca_condition"] = plot_pca_by_condition(
        pca_df,
        out_path=out_dir / "PCA_condition.png",
        palette=pal,
        group_col="group",
        explained_variance_ratio=evr,
        show=show,
    )
    paths["pca_age"] = plot_pca_by_age(
        pca_df,
        age_by_sample=age_ser,
        out_path=out_dir / "PCA_age.png",
        explained_variance_ratio=evr,
        show=show,
    )
    return paths

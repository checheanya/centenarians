"""Plots: theme, PCA, correlation heatmap, clinical EDA panels."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# Control vs Centenarian: dark green-teal (Control) vs dark red (Centenarian) for all condition plots.
DEFAULT_CONDITION_PALETTE: dict[str, str] = {
    "Control": "#165f4a",
    "Centenarian": "#8b1c3a",
}


def _filter_valid_rcparams(params: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in params.items() if k in plt.rcParams}


def _publication_rcparams() -> dict[str, Any]:
    return {
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "savefig.facecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Arial",
            "Helvetica Neue",
            "Helvetica",
            "DejaVu Sans",
            "Liberation Sans",
            "sans-serif",
        ],
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "axes.linewidth": 0.9,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "lines.linewidth": 1.25,
        "lines.markersize": 6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "0.2",
        "text.color": "0.15",
        "axes.labelcolor": "0.15",
        "xtick.color": "0.2",
        "ytick.color": "0.2",
        "axes.titleweight": "normal",
        "axes.titlepad": 8,
        "axes.labelpad": 4,
    }


def _set_theme_safe(
    *,
    style: str,
    context: str,
    font_scale: float,
    palette: str,
) -> None:
    try:
        sns.set_theme(style=style, context=context, font_scale=font_scale, palette=palette)
    except (ValueError, TypeError):
        sns.set_theme(style=style, context=context, font_scale=font_scale)
        try:
            sns.set_palette(palette)
        except (ValueError, TypeError):
            sns.set_palette("deep")


def configure_plot_style(
    *,
    preset: Literal["publication", "notebook"] = "publication",
    max_columns: int = 60,
    font_scale: float = 1.0,
    palette: str = "colorblind",
) -> None:
    pd.set_option("display.max_columns", max_columns)

    if preset == "publication":
        _set_theme_safe(
            style="ticks",
            context="paper",
            font_scale=font_scale * 1.12,
            palette=palette,
        )
        plt.rcParams.update(_filter_valid_rcparams(_publication_rcparams()))
        extra = _filter_valid_rcparams({"legend.title_fontsize": 9})
        plt.rcParams.update(extra)
    else:
        _set_theme_safe(
            style="whitegrid",
            context="notebook",
            font_scale=font_scale,
            palette=palette,
        )
        plt.rcParams.update(
            _filter_valid_rcparams(
                {
                    "figure.dpi": 100,
                    "savefig.dpi": 150,
                    "savefig.bbox": "tight",
                    "legend.frameon": True,
                }
            )
        )


def _palette_color(i: int) -> Any:
    pal = sns.color_palette()
    return pal[i % len(pal)]


def _show_figure_if_requested(fig: plt.Figure, show: bool) -> None:
    if show:
        plt.show()


def plot_sample_pca(
    pca_df: pd.DataFrame,
    *,
    x: str = "PC1",
    y: str = "PC2",
    label_column: str | None = "sample",
    color_column: str | None = None,
    palette: dict[str, str] | None = None,
    figsize: tuple[float, float] = (7, 6),
    point_size: int = 90,
    label_fontsize: int = 8,
    title: str = "PCA of samples",
    edgecolor: str = "white",
    edgewidth: float = 0.7,
    show: bool = False,
) -> plt.Figure:
    """Scatter PC1 vs PC2; optional hue from ``color_column`` and optional ``palette`` dict for group colors."""
    fig, ax = plt.subplots(figsize=figsize)
    scatter_kwargs: dict[str, Any] = dict(
        data=pca_df,
        x=x,
        y=y,
        s=point_size,
        ax=ax,
        edgecolor=edgecolor,
        linewidth=edgewidth,
    )

    if color_column is not None:
        scatter_kwargs["hue"] = color_column
        if palette is not None:
            scatter_kwargs["palette"] = palette
            scatter_kwargs["hue_order"] = list(palette.keys())
        sns.scatterplot(**scatter_kwargs)
        leg = ax.get_legend()
        if leg is not None:
            leg.set_title(color_column)
    else:
        scatter_kwargs["color"] = _palette_color(0)
        sns.scatterplot(**scatter_kwargs)
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

    if label_column is not None:
        for _, row in pca_df.iterrows():
            ax.text(
                row[x],
                row[y],
                str(row[label_column]),
                fontsize=label_fontsize,
                ha="left",
                va="bottom",
                clip_on=False,
            )

    ax.set_title(title)
    sns.despine(ax=ax)
    plt.tight_layout()
    _show_figure_if_requested(fig, show)
    return fig


def plot_sample_correlation_heatmap(
    corr: pd.DataFrame,
    *,
    figsize: tuple[float, float] = (12, 10),
    title: str = "Sample-sample Spearman correlation",
    cmap: str = "vlag",
    center: float = 0,
    square: bool = True,
    cbar_label: str = "Correlation",
    show: bool = False,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        corr,
        cmap=cmap,
        center=center,
        square=square,
        linewidths=0.4,
        linecolor="0.95",
        ax=ax,
        cbar_kws={"label": cbar_label, "shrink": 0.82},
        vmin=-1,
        vmax=1,
    )
    ax.set_title(title)
    plt.tight_layout()
    _show_figure_if_requested(fig, show)
    return fig


# --- Clinical EDA (Control vs Centenarian; age on x-axis) ---


def _format_p_value(p: float) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    if p < 1e-4:
        return "p < 0.0001"
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.3g}"


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


def _clinical_present(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def _filter_two_groups(
    df: pd.DataFrame,
    condition_col: str,
    allowed: tuple[str, ...] = ("Control", "Centenarian"),
) -> pd.DataFrame:
    s = df[condition_col].astype(str).str.strip()
    return df.loc[s.isin(allowed)].copy()


def _clinical_subplots_grid(
    n: int, ncols: int, figsize_per: tuple[float, float]
) -> tuple[plt.Figure, np.ndarray]:
    if n == 0:
        raise ValueError("No columns to plot")
    nrows = int(math.ceil(n / ncols))
    fig_w, fig_h = figsize_per
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * fig_w, nrows * fig_h),
        squeeze=False,
    )
    return fig, axes.ravel()


def plot_condition_boxplot_panels(
    df: pd.DataFrame,
    columns: list[str],
    *,
    condition_col: str = "Condition",
    palette: dict[str, str],
    title: str,
    out_path: Path,
    ncols: int = 4,
    figsize_per: tuple[float, float] = (3.4, 2.9),
    order: tuple[str, ...] = ("Control", "Centenarian"),
    continuous_test: Literal["mannwhitney", "ttest"] = "mannwhitney",
    show_group_stats: bool = True,
    show: bool = False,
) -> Path:
    """Grid of boxplots (or countplots) by condition; saves PNG to ``out_path``.

    X-axis labels for categorical group axes include ``(n=…)`` per category.
    Continuous outcomes: two-group **Mann–Whitney U** (default) or **Welch t-test**.
    Categorical count plots (e.g. Gender × Condition): **χ²** test of independence.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cols = _clinical_present(df, columns)
    if not cols:
        raise ValueError("None of the requested columns exist in the dataframe")

    d = _filter_two_groups(df, condition_col)
    plot_cols: list[tuple[str, str]] = []
    for c in cols:
        if c == condition_col:
            plot_cols.append((c, "counts"))
        elif c == "Gender" or (d[c].dtype == object and d[c].nunique(dropna=True) <= 6):
            plot_cols.append((c, "count_hue"))
        else:
            plot_cols.append((c, "box"))

    n = len(plot_cols)
    fig, axes = _clinical_subplots_grid(n, ncols, figsize_per)

    for ax, (col, kind) in zip(axes, plot_cols):
        if kind == "counts":
            vc = d[condition_col].value_counts().reindex(list(order)).fillna(0)
            colors = [palette.get(str(x), "#888888") for x in vc.index]
            ax.bar(range(len(vc)), vc.values, color=colors, edgecolor="0.3", linewidth=0.5)
            ax.set_xticks(range(len(vc)))
            if show_group_stats:
                xlabs = [f"{str(x)}\n(n={int(v)})" for x, v in zip(vc.index, vc.values)]
            else:
                xlabs = [str(x) for x in vc.index]
            ax.set_xticklabels(xlabs, rotation=15, ha="right")
            ax.set_ylabel("N")
            ax.set_title("N by group")
        elif kind == "count_hue":
            sub = d[[condition_col, col]].dropna()
            x_order = sorted(sub[col].dropna().unique(), key=lambda x: str(x))
            sns.countplot(
                data=sub,
                x=col,
                hue=condition_col,
                ax=ax,
                palette=palette,
                hue_order=list(order),
                order=x_order,
            )
            if show_group_stats:
                labs = [
                    f"{xc}\n(n={int((sub[col] == xc).sum())})" for xc in x_order
                ]
                ax.set_xticks(range(len(x_order)))
                ax.set_xticklabels(labs, rotation=25, ha="right")
            else:
                ax.tick_params(axis="x", rotation=25)
            p_cat = _chi2_independence_p(sub, col, condition_col)
            stat_line = (
                f"χ² {_format_p_value(p_cat)}" if _format_p_value(p_cat) else ""
            )
            ax.set_title(f"{col}\n{stat_line}" if stat_line else col, fontsize=9)
            ax.legend(title=condition_col, frameon=False, fontsize=8)
        else:
            sub = d[[condition_col, col]].copy()
            sub[col] = pd.to_numeric(sub[col], errors="coerce")
            sub = sub.dropna(subset=[col])
            if sub.empty:
                ax.set_title(f"{col} (no numeric data)")
                ax.axis("off")
                continue
            sns.boxplot(
                data=sub,
                x=condition_col,
                y=col,
                hue=condition_col,
                ax=ax,
                palette=palette,
                order=list(order),
                hue_order=list(order),
                dodge=False,
                width=0.55,
                fliersize=2,
                linewidth=0.9,
            )
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()
            g0 = sub[sub[condition_col].astype(str).str.strip() == order[0]][col].values
            g1 = sub[sub[condition_col].astype(str).str.strip() == order[1]][col].values
            p_val = _two_group_continuous_p(g0, g1, method=continuous_test)
            test_name = "MWU" if continuous_test == "mannwhitney" else "Welch t"
            stat_line = (
                f"{test_name} {_format_p_value(p_val)}" if _format_p_value(p_val) else ""
            )
            ax.set_title(f"{col}\n{stat_line}" if stat_line else col, fontsize=9)
            ax.set_xlabel("")
            if show_group_stats:
                n_per = {
                    g: int(
                        (sub[condition_col].astype(str).str.strip() == g).sum()
                    )
                    for g in order
                }
                ax.set_xticks(range(len(order)))
                ax.set_xticklabels(
                    [f"{g}\n(n={n_per[g]})" for g in order],
                    rotation=0,
                    ha="center",
                )

    for j in range(len(plot_cols), len(axes)):
        axes[j].axis("off")

    fig.suptitle(title, fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def plot_age_scatter_panels(
    df: pd.DataFrame,
    columns: list[str],
    *,
    age_col: str = "Age",
    condition_col: str = "Condition",
    palette: dict[str, str],
    title: str,
    out_path: Path,
    ncols: int = 4,
    figsize_per: tuple[float, float] = (3.4, 2.9),
    order: tuple[str, ...] = ("Control", "Centenarian"),
    show: bool = False,
) -> Path:
    """Grid of scatter plots: Age vs each column, colored by condition."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cols = [c for c in _clinical_present(df, columns) if c != age_col and c != condition_col]
    if age_col not in df.columns:
        raise ValueError(f"Age column {age_col!r} not in dataframe")

    d = _filter_two_groups(df, condition_col)
    d = d.copy()
    d[age_col] = pd.to_numeric(d[age_col], errors="coerce")

    if not cols:
        raise ValueError("No y-columns for age plots")

    n = len(cols)
    fig, axes = _clinical_subplots_grid(n, ncols, figsize_per)

    for ax, col in zip(axes, cols):
        sub = d[[age_col, condition_col, col]].copy()
        sub[col] = pd.to_numeric(sub[col], errors="coerce")
        sub = sub.dropna(subset=[age_col, col])
        if sub.empty:
            ax.set_title(f"{col} (no data)")
            ax.axis("off")
            continue
        for grp in order:
            m = sub[sub[condition_col].astype(str).str.strip() == grp]
            if m.empty:
                continue
            ax.scatter(
                m[age_col],
                m[col],
                c=palette.get(grp, "#333333"),
                label=grp,
                alpha=0.65,
                s=18,
                edgecolors="white",
                linewidths=0.3,
            )
        ax.set_xlabel("Age")
        ax.set_ylabel(col)
        ax.set_title(col)
        ax.legend(frameon=False, fontsize=7, loc="best")

    for j in range(len(cols), len(axes)):
        axes[j].axis("off")

    fig.suptitle(title, fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def run_clinical_eda_exports(
    meta_full: pd.DataFrame,
    *,
    to_demographic: list[str],
    to_clinical: list[str],
    to_blood: list[str],
    palette: dict[str, str],
    out_dir: str | Path,
    condition_col: str = "Condition",
    age_col: str = "Age",
    continuous_test: Literal["mannwhitney", "ttest"] = "mannwhitney",
    show_group_stats: bool = True,
    export_univariate_table: bool = True,
    univariate_data_dir: str | Path | None = None,
    show: bool = False,
) -> dict[str, Path]:
    """Save six figures (3 categories × box + age-scatter) under ``out_dir``.

    Box/categorical panels: group **n** on x-axis labels; continuous outcomes use
    Mann–Whitney U or Welch *t*; categorical count plots use χ² (see ``plot_condition_boxplot_panels``).

    If ``export_univariate_table`` is True, also writes CSV + summary under
    ``results/data/EDA_clinical/`` (derived from ``out_dir`` unless ``univariate_data_dir`` is set)
    and prints the full table and FDR summary.
    """
    from .clinical_stats import run_clinical_univariate_export

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    if export_univariate_table:
        if univariate_data_dir is None:
            univariate_data_dir = Path(out_dir).resolve().parents[1] / "data" / "EDA_clinical"
        else:
            univariate_data_dir = Path(univariate_data_dir)
        univariate_data_dir.mkdir(parents=True, exist_ok=True)
        csv_path = univariate_data_dir / "clinical_univariate_tests.csv"
        txt_path = univariate_data_dir / "clinical_univariate_significant_summary.txt"
        run_clinical_univariate_export(
            meta_full,
            to_demographic=to_demographic,
            to_clinical=to_clinical,
            to_blood=to_blood,
            condition_col=condition_col,
            continuous_test=continuous_test,
            out_csv=csv_path,
            out_summary=txt_path,
            print_table=True,
        )
        paths["univariate_tests_csv"] = csv_path
        paths["univariate_summary_txt"] = txt_path

    def go(name: str, cols: list[str]) -> None:
        if not cols:
            return
        p1 = out_dir / f"{name}_condition_boxplots.png"
        p2 = out_dir / f"{name}_age_scatter.png"
        plot_condition_boxplot_panels(
            meta_full,
            cols,
            condition_col=condition_col,
            palette=palette,
            title=f"{name.replace('_', ' ').title()} — by {condition_col}",
            out_path=p1,
            continuous_test=continuous_test,
            show_group_stats=show_group_stats,
            show=show,
        )
        plot_age_scatter_panels(
            meta_full,
            cols,
            age_col=age_col,
            condition_col=condition_col,
            palette=palette,
            title=f"{name.replace('_', ' ').title()} — vs {age_col}",
            out_path=p2,
            show=show,
        )
        paths[f"{name}_box"] = p1
        paths[f"{name}_age"] = p2

    go("demographic", to_demographic)
    go("clinical", to_clinical)
    go("blood", to_blood)
    return paths


def save_figure(
    fig: plt.Figure,
    path: str | Path,
    *,
    dpi: int = 300,
    transparent: bool = False,
    show: bool = False,
    **kwargs: Any,
) -> Path:
    """
    Save a figure for publication (default 300 DPI, tight bounding box).

    Use vector formats when journals allow: ``path`` ending in ``.pdf`` or ``.svg``.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    face = "none" if transparent else "white"
    fig.savefig(
        p,
        dpi=dpi,
        bbox_inches="tight",
        facecolor=face,
        edgecolor="none",
        **kwargs,
    )
    _show_figure_if_requested(fig, show)
    return p

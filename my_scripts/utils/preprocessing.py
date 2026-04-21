"""Proteomics: load, matrix build, transforms, PCA, correlation, differential abundance."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

DEFAULT_PROTEIN_ID_COLUMNS = (
    "PG.ProteinGroups",
    "PG.Genes",
    "PG.ProteinDescriptions",
    "PG.ProteinNames",
)


def load_proteomics_tsv(
    path: str | Path,
    *,
    sep: str = "\t",
    **read_csv_kwargs: Any,
) -> pd.DataFrame:
    """Load a proteomics report TSV (or CSV)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    kw = {"sep": sep, "low_memory": False}
    kw.update(read_csv_kwargs)
    return pd.read_csv(p, **kw)


def detect_abundance_columns(
    columns: pd.Index | list[str],
    *,
    suffix: str = ".PG.Quantity",
) -> tuple[list[str], list[str]]:
    """Split column names into abundance vs metadata using a suffix (default: Spectronaut-style)."""
    cols = list(columns)
    abundance = [c for c in cols if str(c).endswith(suffix)]
    meta = [c for c in cols if c not in abundance]
    return abundance, meta


def resolve_protein_id_columns(
    df: pd.DataFrame,
    preferred: tuple[str, ...] | list[str] | None = None,
) -> list[str]:
    """Return id columns that exist in ``df`` (order preserved)."""
    pref = list(preferred) if preferred is not None else list(DEFAULT_PROTEIN_ID_COLUMNS)
    return [c for c in pref if c in df.columns]


def build_protein_matrix(
    df: pd.DataFrame,
    abundance_cols: list[str],
    *,
    id_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Subset to id + abundance columns and coerce abundance to numeric."""
    ids = resolve_protein_id_columns(df) if id_cols is None else id_cols
    out = df[ids + abundance_cols].copy()
    out[abundance_cols] = out[abundance_cols].apply(pd.to_numeric, errors="coerce")
    return out


def log2_abundance(
    X: pd.DataFrame,
    *,
    zero_as_nan: bool = True,
) -> pd.DataFrame:
    """Log2 transform; zeros optionally become NaN before log."""
    out = X.copy()
    if zero_as_nan:
        out = out.replace(0, np.nan)
    return np.log2(out)


def median_center_samples(X_log2: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Subtract each sample's median (column-wise) for median normalization."""
    medians = X_log2.median(axis=0, skipna=True)
    return X_log2 - medians, medians


def sample_pca(
    abundance_samples_as_columns: pd.DataFrame,
    *,
    n_components: int = 2,
    random_state: int | None = 42,
    palette: dict | None = None,
    sample_to_group: dict | None = None,
) -> tuple[pd.DataFrame, PCA, np.ndarray]:
    """
    PCA with samples as observations (columns = samples in input matrix).

    Rows = proteins/features, columns = samples. Missing values are median-imputed per protein,
    then features are standardized before PCA.

    Optionally pass ``sample_to_group`` and ``palette`` to attach group/color columns to the result.
    """
    X = abundance_samples_as_columns.T
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_scaled = scaler.fit_transform(X_imp)
    pca = PCA(n_components=n_components, random_state=random_state)
    pcs = pca.fit_transform(X_scaled)
    names = [f"PC{i + 1}" for i in range(n_components)]
    pca_df = pd.DataFrame(pcs, columns=names)
    pca_df.insert(0, "sample", X.index)

    if sample_to_group is not None:
        group_ser = pd.Series(sample_to_group)
        pca_df["group"] = pca_df["sample"].map(group_ser)
        if palette is not None:
            pca_df["color"] = pca_df["group"].map(palette)

    return pca_df, pca, pca.explained_variance_ratio_


def sample_correlation_matrix(
    X: pd.DataFrame,
    *,
    method: str = "spearman",
) -> pd.DataFrame:
    """Pairwise correlation between samples (columns)."""
    return X.corr(method=method)


def two_group_differential_abundance(
    X_log2_norm: pd.DataFrame,
    id_table: pd.DataFrame,
    sample_to_group: dict[str, str],
    *,
    min_per_group: int = 2,
) -> pd.DataFrame | None:
    """
    Welch t-test per protein; log2FC = mean(first label) - mean(second label) in
    ``pd.Series(sample_to_group).unique()`` order.
    """
    grouped = pd.Series(sample_to_group)
    valid = grouped.index.intersection(X_log2_norm.columns)
    if len(valid) == 0:
        return None
    y = grouped.loc[valid]
    groups = y.unique()
    if len(groups) != 2:
        raise ValueError(f"Expected exactly 2 groups, got {len(groups)}: {groups!r}")
    g1, g2 = groups[0], groups[1]
    s1 = y[y == g1].index.tolist()
    s2 = y[y == g2].index.tolist()
    mat = X_log2_norm[valid]

    pvals: list[float] = []
    log2fc: list[float] = []
    for _, row in mat.iterrows():
        a = row[s1].dropna().values
        b = row[s2].dropna().values
        if len(a) < min_per_group or len(b) < min_per_group:
            pvals.append(np.nan)
            log2fc.append(np.nan)
            continue
        _, p = ttest_ind(a, b, equal_var=False)
        pvals.append(float(p))
        log2fc.append(float(np.nanmean(a) - np.nanmean(b)))

    res = id_table.copy()
    res["log2FC"] = log2fc
    res["pvalue"] = pvals
    mask = res["pvalue"].notna()
    if mask.any():
        res.loc[mask, "padj"] = multipletests(res.loc[mask, "pvalue"], method="fdr_bh")[1]
    else:
        res["padj"] = np.nan
    return res.sort_values(["padj", "pvalue"], na_position="last")

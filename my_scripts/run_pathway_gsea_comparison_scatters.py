#!/usr/bin/env python3
"""
Build KEGG prerank GSEA for UKBB (aging rank) and each cohort (CIAO, Swiss100, NECS),
merge on pathway terms, and save NES scatter plots in the style of make_figure3_v3 panel e.

Uses the same library as ``pathway_prerank_dotplot`` (KEGG_2021_Human). Reuses the existing
CIAO prerank table from ``DE_basic_pathway_KEGG_prerank.csv`` when present and ``--reuse``.

  python run_pathway_gsea_comparison_scatters.py --reuse
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# project layout: centenerians/my_scripts/this_file.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "my_scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "my_scripts"))

from utils.pathway_gsea_scatter import (  # noqa: E402
    build_ciao_de_prerank,
    build_necs_prerank,
    build_swiss_prerank,
    build_ukbb_age_prerank,
    run_pair_and_plot,
)
from utils.ukbb_escape import preprocess_ukbb_protein_table  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="UKBB vs cohort KEGG NES scatter plots (Figure 3e style).")
    p.add_argument("--reuse", action="store_true", help="Reuse cached prerank CSVs when present.")
    p.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT,
        help="Project root (default: parent of my_scripts/).",
    )
    args = p.parse_args()
    root: Path = args.root.resolve()

    data_dir = root / "results/data/pathway_gsea_compare"
    plot_dir = root / "results/plots/pathway_gsea_compare"
    data_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    ukbb_csv = root / "sanju_version/1_raw_data/nature2023_age_proteins.csv"
    ukbb = pd.read_csv(ukbb_csv)
    ukbb_pre = preprocess_ukbb_protein_table(ukbb.copy())
    rnk_ukbb = build_ukbb_age_prerank(ukbb_pre)
    cache_ukbb = data_dir / "ukbb_KEGG_prerank.csv"

    de_csv = root / "results/data/DE_basic/DE_basic_full.csv"
    de_full = pd.read_csv(de_csv)
    rnk_ciao = build_ciao_de_prerank(de_full)
    cache_ciao_existing = root / "results/data/DE_basic/DE_basic_pathway_KEGG_prerank.csv"
    cache_ciao = cache_ciao_existing if cache_ciao_existing.is_file() else data_dir / "ciao_KEGG_prerank.csv"

    swiss_xlsx = root / "data/SWISS100/acel70409-sup-0005-tables2.xlsx"
    swiss = pd.read_excel(swiss_xlsx, sheet_name="2.a Centenarian - Healthy", index_col=0)
    swiss["Gene"] = swiss["Assay"]
    rnk_swiss = build_swiss_prerank(swiss)
    cache_swiss = data_dir / "swiss100_KEGG_prerank.csv"

    necs_xlsx = (
        root
        / "data/New_England_Sebastiani/acel13290-sup-0001-appendixs1/acel13290-sup-0002-TableS1.xlsx"
    )
    necs = pd.read_excel(necs_xlsx, sheet_name="Supplement-Table 1a")
    necs = necs.sort_values("Adj.Pvalue").drop_duplicates("geneID", keep="first")
    rnk_necs = build_necs_prerank(necs)
    cache_necs = data_dir / "necs_KEGG_prerank.csv"

    jobs = [
        (
            rnk_ciao,
            cache_ciao,
            data_dir / "gsea_merged_UKBB_vs_CIAO.csv",
            plot_dir / "pathway_nes_UKBB_vs_CIAO.png",
            "NES (UKBB aging)",
            "NES (CIAO centenarian DE)",
            "Pathway GSEA: UKBB aging vs CIAO centenarian",
        ),
        (
            rnk_swiss,
            cache_swiss,
            data_dir / "gsea_merged_UKBB_vs_Swiss100.csv",
            plot_dir / "pathway_nes_UKBB_vs_Swiss100.png",
            "NES (UKBB aging)",
            "NES (Swiss100 centenarian DE)",
            "Pathway GSEA: UKBB aging vs Swiss100",
        ),
        (
            rnk_necs,
            cache_necs,
            data_dir / "gsea_merged_UKBB_vs_NECS.csv",
            plot_dir / "pathway_nes_UKBB_vs_NECS.png",
            "NES (UKBB aging)",
            "NES (NECS log2 FC.cont2cent rank)",
            "Pathway GSEA: UKBB aging vs NECS (Table S1a)",
        ),
    ]

    want_reuse = bool(args.reuse)
    for rnk_cohort, cache_cohort, merged_csv, plot_png, xlab, ylab, title in jobs:
        cache_cohort = Path(cache_cohort)
        run_pair_and_plot(
            rnk_ukbb,
            rnk_cohort,
            out_merged_csv=merged_csv,
            out_plot=plot_png,
            cache_left_csv=cache_ukbb,
            cache_right_csv=cache_cohort,
            label_left="UKBB",
            label_right="Cohort",
            xlabel=xlab,
            ylabel=ylab,
            title=title,
            reuse_left=want_reuse and cache_ukbb.is_file(),
            reuse_right=want_reuse and cache_cohort.is_file(),
        )
        print(f"Wrote {merged_csv} and {plot_png}")

    print("Done.")


if __name__ == "__main__":
    main()

#! /usr/bin/env python

"""
Analysis of single case of metastatic sarcomatoid tumor with imaging mass cytometry.
"""

import sys
import os
import argparse
from typing import Tuple
from os.path import join as pjoin
from datetime import datetime

from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from skimage.exposure import equalize_hist as eq
import scanpy as sc
import pandas as pd
from sklearn.mixture import GaussianMixture
import seaborn as sns

# from skimage import exposure
from imc.types import Path
from imc import Project
from imc.graphics import get_grid_dims, add_legend
from imc.operations import measure_channel_background, fit_gaussian_mixture


matplotlib.rcParams["svg.fonttype"] = "none"
FIG_KWS = dict(dpi=300, bbox_inches="tight")


CUR_DATE_YYYYMMDD = datetime.today().strftime("%Y-%m-%d")
CUR_DATE_YYYYMMDD = "2020-05-19"
CLI = [
    "metadata/annotation.csv",
    "20200122_PD_L1_100_percent_case",
    "7,8,9,10",
    "processed",
    f"analysis/case_20200122.a{CUR_DATE_YYYYMMDD}/20200122-rois_10-13.",
]
cli = None


def parse_arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(dest="metadata", help="CSV sample metadata annotation.")
    parser.add_argument(
        "--panel-metadata",
        dest="panel_metadata",
        help="CSV file with panel metadata.",
    )
    parser.add_argument(dest="sample", help="Single sample to analyze.")
    parser.add_argument(
        dest="rois", help="ROIs to analyze, must be comma separated. Eg. 1,2,3."
    )
    parser.add_argument(dest="processed_dir", help="Parent dir with processed data.")
    parser.add_argument(dest="output_prefix", help="Prefix for analysis.")

    return parser


def main(cli=None) -> int:
    args = parse_arguments().parse_args(cli or CLI)

    prj = Project(
        sample_metadata=args.metadata,
        processed_dir=args.processed_dir,
        toggle=False,
    )

    sample = [s for s in prj.samples if s.name == args.sample][0]
    sample.rois = [r for r in sample.rois if str(r.roi_number) in args.rois.split(",")]
    prj.samples = [sample]

    output_prefix = Path(args.output_prefix)
    output_dir = os.path.dirname(args.output_prefix)
    os.makedirs(output_dir, exist_ok=True)

    # QC
    # # Plot channel means
    channel_labels = sample.rois[0].channel_labels
    # for marker in ['mean', 'DNA', 'PDL1', 'Keratin', 'CD45(']:
    for marker in tqdm(channel_labels):
        fig = sample.plot_rois(marker)
        fig.suptitle(f"{sample}")
        fig.savefig(output_prefix + f"all_rois.{marker}.pdf", **FIG_KWS)
        plt.close("all")

    # # Plot all channels for each roi separately
    for roi in tqdm(sample.rois):
        plot_file = output_prefix + f"{roi.name}.all_channels.pdf"
        fig = roi.plot_channels()
        fig.savefig(plot_file, **FIG_KWS)
        plt.close("all")

    # # Plot segmentation
    for sample in prj.samples:
        plot_file = output_prefix + "plot_probabilities_and_segmentation.all_rois.pdf"
        # if os.path.exists(plot_file):
        #     continue
        print(sample)
        sample.read_all_inputs(
            only_these_keys=["probabilities", "cell_mask", "nuclei_mask"]
        )
        fig = sample.plot_probabilities_and_segmentation()
        fig.savefig(plot_file, **FIG_KWS)
        plt.close("all")

    # Image QC
    prj.image_summary()

    # Channel QC
    # fig, corr = prj.channel_correlation()
    scores = measure_channel_background(prj.rois, output_prefix=output_prefix, plot=False)

    # Single cell analysis
    prj.quantify_cells()
    channel_includelist = ["GranzymeB(Er167)", "IDO(Nd144)", "CXCL12(Dy163)"]
    channel_excludelist = scores[
        (scores < 0.11) & ~(scores.index.isin(channel_includelist))
    ].index
    quantification = prj.quantification.drop(channel_excludelist, axis=1)

    prj.cluster_cells(
        output_prefix=output_prefix,
        cell_type_channels=[x for x in channel_labels if "HLA" not in x],
        quantification=quantification,
        leiden_clustering_resolution=0.75,
    )

    # Plot cell type assignments
    fig = sample.plot_cell_types()
    fig.savefig(output_prefix + "cell_types_as_image.svg", **FIG_KWS)

    # # plot a new version with clusters reordered/renamed
    cmap = "tab20c"
    new_cell_types = [
        (1, "1 - CD45mid, CD14+, CD163+, CD206+, CD68+"),
        (2, "2 - CD45+, CD68+"),
        (7, "3 - CD14+, CD68+"),
        (9, "4 - ColTypeI"),
        (3, "5 - PanKeratin+, Vimentin+, PDL1+"),
        (5, "6 - PanKeratin+, Ki67+"),
        (6, "7 - PanKeratin+"),
        (8, "8 - Vimentin+"),
        (12, f"{5 + 20} - PanKeratin+, ECadherin+, PDL1+, CXCL12+"),
        (4, "9 - CD45+, CD3+, CD4+"),
        (10, "10 - CD45+, CD3+, CD8a+"),
        (11, "13 - AlphaSMA+, CD56+"),
        (13, "14 - AlphaSMA+, pHH3+"),
    ]
    clusters = sample.clusters
    new_clusters = clusters.copy()
    for cur_n, new_label in new_cell_types:
        f = clusters.str.startswith(f"{cur_n} - ")
        new_clusters.loc[f] = new_label
    fig = sample.plot_cell_types(cell_type_assignments=new_clusters, palette=cmap)
    fig.savefig(output_prefix + "cell_types_as_image.renamed.svg", **FIG_KWS)

    # # plot a more coarse version
    cmap = "tab10"
    new_cell_types_coarse = [
        (1, "1 - Myeloid"),
        (2, "1 - Myeloid"),
        (7, "1 - Myeloid"),
        (9, "1 - Myeloid"),
        (6, "1 - Myeloid"),
        (3, "2 - Tumor"),
        (5, "2 - Tumor"),
        (8, "2 - Tumor"),
        (12, "2 - Tumor"),
        (4, "3 - T cell"),
        (10, "3 - T cell"),
        (11, "4 - Muscle"),
        (13, "4 - Muscle"),
    ]
    clusters = sample.clusters
    coarse_clusters = clusters.copy()
    for cur_n, new_label in new_cell_types_coarse:
        f = clusters.str.startswith(f"{cur_n} - ")
        coarse_clusters.loc[f] = new_label
    fig = sample.plot_cell_types(cell_type_assignments=coarse_clusters, palette=cmap)
    fig.savefig(output_prefix + "cell_types_as_image.coarse.svg", **FIG_KWS)

    # Clustermap with renamed clusters
    ann = sc.read(output_prefix + "single_cell.processed.h5ad")
    from imc.utils import get_mean_expression_per_cluster, double_z_score

    means = get_mean_expression_per_cluster(ann)

    columns = means.columns.to_series()
    new_columns = columns.copy()
    for cur_n, new_label in new_cell_types:
        f = columns.str.startswith(f"{cur_n} - ")
        new_columns.loc[f] = new_label
    means.columns = new_columns
    grid = sns.clustermap(
        double_z_score(means),
        center=0,
        cmap="RdBu_r",
        robust=True,
        cbar_kws=dict(label="Mean intensity (Z-score)"),
        row_colors=means.mean(1).rename("Channel mean"),
        col_colors=ann.obs["cluster"].value_counts().rename("Cells per cluster"),
    )
    grid.savefig(
        output_prefix + "cell_types.mean_expression.clustermap.double_z_score.svg"
    )

    # Plot some markers
    for roi in sample.rois:
        fig, axes = plt.subplots(4, 2, figsize=(2 * 5, 4 * 5))
        patches = roi.plot_cell_types(
            ax=axes[0, 0, np.newaxis, np.newaxis],
            cell_type_assignments=new_clusters[roi.name],
            palette="tab20c",
        )
        add_legend(
            patches,
            axes.flatten()[0],
            bbox_to_anchor=(-0.05, 1),
            loc=1,
            borderaxespad=0.0,
        )
        for i, ch in enumerate(
            ["DNA", "Keratin", "Vimentin", "CD45(", "CD3(", "CD14", "CD68"]
        ):
            roi.plot_channel(ch, ax=axes.flatten()[i + 1])
        axes.flatten()[-1].axis("off")
        plot_file = output_prefix + f"{roi.name}.cell_types.markers.pdf"
        fig.savefig(plot_file, **FIG_KWS)
        plt.close("all")

    # Community

    return 0


def analysis(sample, output_prefix, raw=False):
    h5ad_file = output_prefix + "single_cell.processed.h5ad"
    a = sc.read(h5ad_file)
    a.obs.index = a.obs.index.astype(int)
    if raw:
        expr_norm = pd.DataFrame(
            a.raw.X, index=a.obs.index, columns=a.var.index
        ).reset_index(drop=True)
        expr_norm = np.log1p(expr_norm)
    else:
        expr_norm = pd.DataFrame(a.X, index=a.obs.index, columns=a.var.index).reset_index(
            drop=True
        )

    expr_norm.to_csv(output_prefix + "expression.normalized.csv")
    clusters = a.obs["cluster"].astype(str)

    # # reordered/renamed clusters
    new_cell_types = [
        (1, "1 - CD45mid, CD14+, CD163+, CD206+, CD68+"),
        (2, "2 - CD45+, CD68+"),
        (7, "3 - CD14+, CD68+"),
        (9, "4 - ColTypeI"),
        (3, "5 - PanKeratin+, Vimentin+, PDL1+"),
        (5, "6 - PanKeratin+, Ki67+"),
        (6, "7 - PanKeratin+"),
        (8, "8 - Vimentin+"),
        (12, f"{5 + 20} - PanKeratin+, ECadherin+, PDL1+, CXCL12+"),
        (4, "9 - CD45+, CD3+, CD4+"),
        (10, "10 - CD45+, CD3+, CD8a+"),
        (11, "13 - AlphaSMA+, CD56+"),
        (13, "14 - AlphaSMA+, pHH3+"),
    ]
    new_clusters = sample.clusters.copy()
    for cur_n, new_label in new_cell_types:
        f = sample.clusters.str.startswith(f"{cur_n} - ")
        new_clusters.loc[f] = new_label
    sample.set_clusters(new_clusters)

    new_clusters = clusters.copy()
    for cur_n, new_label in new_cell_types:
        f = clusters.str.startswith(f"{cur_n} - ")
        new_clusters.loc[f] = new_label
    clusters = new_clusters

    # # declare cells positive for one marker

    N_MIXTURES = 2
    expr_thresh = fit_gaussian_mixture(expr_norm, N_MIXTURES)
    expr_thresh.to_csv(output_prefix + f"expression.{N_MIXTURES}.thresholded.csv")

    expr_thresh = pd.read_csv(
        output_prefix + f"expression.{N_MIXTURES}.thresholded.csv", index_col=0
    )
    # # # # violin
    n, m = get_grid_dims(expr_thresh.shape[1])
    fig, axes = plt.subplots(
        n, m, figsize=(m * 3, n * 3), gridspec_kw=dict(wspace=0, hspace=0)
    )
    axes = axes.flatten()
    for i, (ax, ch) in enumerate(zip(axes, expr_norm.columns)):
        q = expr_norm[ch].to_frame("Expr").join(expr_thresh[ch].rename("Gate"))
        sns.violinplot(x="Gate", y="Expr", data=q, ax=ax)
        ax.text(0.5, expr_norm[ch].max(), s=ch, ha="center", va="top")
    for ax in axes[i + 1 :]:
        ax.axis("off")
        ax.axhline(0, linestyle="--", color="grey")
    fig.savefig(
        output_prefix + f"expression.{N_MIXTURES}.threshold_distributions.violinplot.svg",
        **FIG_KWS,
    )

    pos = expr_thresh.values.max()
    marker_positiveness = (
        ((expr_thresh == pos).sum() / expr_thresh.shape[0]) * 100
    ).sort_values(ascending=False)

    fig, ax = plt.subplots(1, 1, figsize=(3, 7))
    sns.barplot(
        marker_positiveness,
        marker_positiveness.index,
        orient="horiz",
        palette="magma_r",
        ax=ax,
    )
    ax.set_xlabel("% positive cells")
    fig.savefig(output_prefix + "marker_positiveness.svg", **FIG_KWS)

    # cell type assignments
    # clusters = a.obs[["roi", "obj_id", "cluster"]].set_index(["roi", "obj_id"])["cluster"]
    (clusters.value_counts() / clusters.shape[0]) * 100

    fig = sample.plot_cell_types(palette="tab20c")
    fig.savefig(output_prefix + "cell_type_assignments.svg", **FIG_KWS)

    # s = sample.measure_cell_shape()
    # s.to_csv(pjoin(sample.root_dir, "single_cell", "cell_shape.csv"))
    # s = s.assign(cluster=sample.cell_type_assignments.values)

    markers = ["PDL1", "Keratin", r"CD45\("]
    markers = ["PDL1(Lu175)", "PanKeratin(Dy164)", "CD45(Sm152)"]
    # plot separately
    n = len(sample.rois)
    m = len(markers)
    fig, axes = plt.subplots(n, m, figsize=(m * 4, n * 4))
    for i, roi in enumerate(sample.rois):
        roi.plot_channels(markers, axes=axes[i])
    fig.savefig(output_prefix + "PanKeratin_CD45_PDL1_overlap.all_rois.svg", **FIG_KWS)

    # plot expression of markers across clusters
    p = expr_norm[markers].join(clusters).sort_values("cluster")

    fig, ax = plt.subplots(1, 1, figsize=(6, 8))
    ax = sns.boxplot(
        data=p.melt(id_vars="cluster"),
        orient="horiz",
        y="variable",
        x="value",
        hue="cluster",
        ax=ax,
        whis=1e5,
    )
    ax.axvline(0, linestyle="--", color="grey")
    fig.savefig(output_prefix + "PD1_expression.tumour_vs_immune.boxplot.svg", **FIG_KWS)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax = sns.violinplot(
        data=p.melt(id_vars="cluster"),
        orient="horiz",
        y="variable",
        x="value",
        hue="cluster",
        ax=ax,
    )
    ax.axvline(0, linestyle="--", color="grey")
    fig.savefig(
        output_prefix + "PD1_expression.tumour_vs_immune.violinplot.svg",
        **FIG_KWS,
    )

    # Stats
    pos = expr_thresh.values.max()
    markers = [
        "PDL1(Lu175)",
        "PanKeratin(Dy164)",
        "CD45(Sm152)",
        "CD3(Er170)",
        "CD8a(Dy162)",
        "CD4(Gd156)",
        "CD68(Tb159)",
    ]
    # PDL1 on tumour (using GM)
    pdl1_l = markers[0]
    tumor_l = markers[1]
    immune_l = markers[2]

    # Simply print fraction of PDL1+ cells in tumour and immune compartments
    for cell_l in [tumor_l, immune_l]:
        # # fraction of PDL1_l+
        c = expr_thresh.loc[expr_thresh[cell_l] == pos]
        print(cell_l, (c.loc[:, pdl1_l].value_counts()[pos] / c.shape[0]) * 100)
        # expr_norm.loc[c.index, pdl1_l]  <- these would be the cells

    # expression values of PDL1 on tumor
    # expr_norm = expr_norm.loc[
    #     expr_norm[markers].mean(1) > expr_norm[markers].mean(1).quantile(0.001)
    # ]
    # expr_norm = expr_norm.loc[(expr_norm[markers] > 100).all(1)]
    expr_norm += 1

    tumor = expr_thresh[tumor_l] == pos
    immune = expr_thresh[immune_l] == pos
    tcell = expr_thresh[markers[3]] == pos
    tcell8 = expr_thresh[markers[4]] == pos
    tcell4 = expr_thresh[markers[5]] == pos
    mac = expr_thresh[markers[6]] == pos
    x1 = expr_norm.loc[tumor & ~immune, pdl1_l]
    x2 = expr_norm.loc[immune & ~tumor, pdl1_l]
    x3 = expr_norm.loc[~immune & ~tumor, pdl1_l]
    x4 = expr_norm.loc[immune & tcell, pdl1_l]
    x5 = expr_norm.loc[immune & tcell8, pdl1_l]
    x6 = expr_norm.loc[immune & tcell4, pdl1_l]
    x7 = expr_norm.loc[immune & mac, pdl1_l]
    fig, axes = plt.subplots(1, 2, figsize=(2 * 4, 4))
    sns.distplot(x1, ax=axes[0], label="PanKeratin+, CD45-")
    sns.distplot(x2, ax=axes[0], label="CD45+, PanKeratin-")
    sns.distplot(x3, ax=axes[0], label="PanKeratin-, CD45-")
    sns.distplot(x1, ax=axes[1], label="PanKeratin+, CD45-")
    sns.distplot(x4, ax=axes[1], label="CD45+, CD3+")
    sns.distplot(x5, ax=axes[1], label="CD45+, CD3+, CD8a+")
    sns.distplot(x6, ax=axes[1], label="CD45+, CD3+, CD4+")
    sns.distplot(x7, ax=axes[1], label="CD45+, CD68+")
    for ax in axes:
        ax.set_xlabel("PDL1 expression")
        ax.set_ylabel("Fraction of cells")
        ax.set_xlim(right=expr_norm[pdl1_l].quantile(0.999))
        ax.legend()
    fig.savefig(
        output_prefix + "PDL1_expression.tumor_vs_immune.distribution.distplot.svg",
        **FIG_KWS,
    )

    pdl1 = expr_thresh[pdl1_l] == pos
    x1 = expr_norm.loc[pdl1, markers[1]]
    x2 = expr_norm.loc[~pdl1, markers[1]]
    x3 = expr_norm.loc[pdl1, markers[2]]
    x4 = expr_norm.loc[~pdl1, markers[2]]
    fig, axes = plt.subplots(1, 2, figsize=(2 * 4, 4))
    sns.distplot(x1, ax=axes[0], label="PDL1+")
    sns.distplot(x2, ax=axes[0], label="PDL1-")
    sns.distplot(x3, ax=axes[1], label="PDL1+")
    sns.distplot(x4, ax=axes[1], label="PDL1-")
    axes[0].set_xlabel("PanKeratin expression")
    axes[1].set_xlabel("CD45 expression")
    axes[0].set_xlim(right=expr_norm[markers[1]].quantile(0.999))
    axes[1].set_xlim(right=expr_norm[markers[2]].quantile(0.999))
    for ax in axes:
        ax.set_ylabel("Fraction of cells")
        ax.legend()
    fig.savefig(
        output_prefix
        + "PanKeratin_CD45_expression.tumor_vs_immune.PDL1_distribution.distplot.svg",
        **FIG_KWS,
    )

    # plt.scatter(x=expr_norm[markers[1]], y=expr_norm[markers[2]], c=expr_norm[pdl1_l])

    # Final plots summarizing
    from imc.utils import get_threshold_from_gaussian_mixture

    expr_norm_cluster = expr_norm.join(clusters)
    # threshold = 0.86
    threshold = get_threshold_from_gaussian_mixture(
        x=expr_norm[pdl1_l], y=expr_thresh[pdl1_l]
    )

    pdl1_percentages = (
        (
            expr_norm_cluster[[pdl1_l, "cluster"]]
            .groupby("cluster")[pdl1_l]
            .apply(lambda x: ((x > threshold).sum() / x.shape[0]) * 100)
        )
        .sort_values(ascending=False)
        .rename(r"% PD-L1 positive")
    )
    counts = expr_norm_cluster["cluster"].value_counts().rename("Cell number")
    percents = ((counts / expr_norm_cluster["cluster"].shape[0]) * 100).rename(
        r"% of total cells"
    )

    # to_rep = sample.cell_type_assignments['cluster'].str.extract(r"(\d+) - \d+ - (.*)").drop_duplicates().set_index(0)[1].to_dict()

    joint = (
        counts.to_frame()
        .join(percents)
        .join(pdl1_percentages)
        .sort_values(r"% PD-L1 positive", ascending=False)
        .rename_axis(index="Cluster")
    )
    # joint.index = joint.index.to_series().replace(to_rep).rename("Cluster")
    joint = joint.loc[joint["Cell number"] / joint["Cell number"].sum() >= 0.01]
    to_plot_cat = (
        joint.loc[joint.index.str.contains(r"PanKeratin|CD4\+|CD8a|CD68\+|CD20\+")]
        .reset_index()
        .melt(id_vars="Cluster")
    )

    grid = sns.catplot(
        data=to_plot_cat,
        y="Cluster",
        x="value",
        col="variable",
        palette="Set1",
        sharex=False,
        order=to_plot_cat["Cluster"].drop_duplicates(),
        kind="bar",
        margin_titles=True,
        orient="horiz",
        height=3,
    )
    grid.savefig(
        output_prefix + "PDL1_expression_quantification.barplot.svg",
        **FIG_KWS,
    )

    to_plot = expr_norm_cluster[[pdl1_l, "cluster"]].melt(id_vars="cluster")
    # to_plot["Cluster"] = to_plot["cluster"].replace(to_rep)
    to_plot = to_plot.loc[to_plot["cluster"].isin(to_plot_cat["Cluster"])]

    to_plot["value"] = np.log1p(to_plot["value"])

    for plot_type in ["box", "boxen", "violin"]:
        grid = sns.catplot(
            data=to_plot,
            y="cluster",
            x="value",
            col="variable",
            palette="Set1",
            sharex=False,
            order=to_plot_cat["Cluster"].drop_duplicates(),
            kind=plot_type,
            margin_titles=True,
            orient="horiz",
            height=3,
        )
        grid.ax.axvline(np.log1p(threshold), linestyle="--", color="grey")
        grid.savefig(
            output_prefix + f"PDL1_expression_quantification.{plot_type}plot.svg",
            **FIG_KWS,
        )


def visualizations(sample):
    # Plot image vs segmentation coloured by positivity
    Coord = Tuple[int, int, int, int]

    def plot_overlaied_channels(roi, marker_combs, coords: Coord = None):
        def minmax_scale(x):
            return (x - x.min()) / (x.max() - x.min())

        shape = roi.shape[1], roi.shape[2]
        maskx = np.repeat(False, shape[0])
        masky = np.repeat(False, shape[1])
        if coords is None:
            maskx = ~maskx
            masky = ~masky
        else:
            # maskx[coords[0]:coords[1]] = True
            # masky[coords[2]: coords[3]] = True
            maskx[coords[2] : coords[3]] = True
            masky[coords[0] : coords[1]] = True

        cmaps = get_rgb_cmaps()
        max_inches = 4
        maskshape = np.empty(shape)[maskx][:, masky].shape
        aspect_ratio = maskshape[1] / maskshape[0]
        n, m = len(marker_combs[0]) + 1, len(marker_combs)
        fig, axes = plt.subplots(
            n,
            m,
            figsize=(m * max_inches, n * max_inches / aspect_ratio),
            gridspec_kw=dict(hspace=0.05, wspace=0.05),
            sharex=True,
            sharey=True,
            squeeze=False,
        )
        for j, sp_markers in enumerate(marker_combs):
            # plot channels individually
            for i, marker in enumerate(sp_markers):
                axes[i, j].set_ylabel(marker)
                axes[i, j].imshow(
                    eq(roi._get_channel(marker)[1].squeeze())[maskx, :][:, masky],
                    cmap=cmaps[i],
                    interpolation="gaussian",
                )
            # plor channel merge
            for k, channels in enumerate([sp_markers]):
                k += 1
                axes[i + k, j].set_ylabel(", ".join(channels))
                # Use RGB channel for real mixture
                _rgb = list()
                for marker in channels:
                    _rgb.append(minmax_scale(roi._get_channel(marker)[1].squeeze()))
                if len(_rgb) == 2:
                    _rgb.append(np.zeros(roi.shape[1:]))
                rgb = np.moveaxis(np.asarray(_rgb), 0, -1)
                axes[i + k, j].imshow(rgb[maskx, :][:, masky], interpolation="gaussian")
        for ax in axes.flatten():
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            # ax.axis("off")
        return fig

    marker_combs = [
        ("DNA1(Ir191)", "Vimentin(Sm154)", "ColTypeI(Tm169)"),
        ("PanKeratin(Dy164)", "PDL1(Lu175)", "CD45(Sm152)"),
        ("PanKeratin(Dy164)", "PDL1(Lu175)", "CD3(Er170)"),
        ("CD3(Er170)", "PDL1(Lu175)", "CD4(Gd156)"),
        ("PanKeratin(Dy164)", "PDL1(Lu175)", "CD4(Gd156)"),
        ("PanKeratin(Dy164)", "PDL1(Lu175)", "CD8a(Dy162)"),
        ("CD3(Er170)", "PDL1(Lu175)", "CD8a(Dy162)"),
        ("PanKeratin(Dy164)", "PDL1(Lu175)", "CD68(Tb159)"),
        ("CD68(Tb159)", "PDL1(Lu175)", "GranzymeB(Er167)"),
        ("CD3(Er170)", "CD8a(Dy162)", "GranzymeB(Er167)"),
        ("PanKeratin(Dy164)", "PDL1(Lu175)", "CD20(Dy161)"),
        ("PanKeratin(Dy164)", "PDL1(Lu175)"),
        ("CD8a(Dy162)", "PDL1(Lu175)"),
        ("CD4(Gd156)", "PDL1(Lu175)"),
        ("CD68(Tb159)", "PDL1(Lu175)"),
        ("CD20(Dy161)", "PDL1(Lu175)"),
    ]

    regions = {
        10: [(389, 600, 274, 429)],
        11: [],
        12: [],
        13: [(463, 729, 16, 157), (239, 317, 126, 174), (372, 573, 57, 240)],
    }

    for roi in sample.rois:
        print(roi)
        f = pjoin(
            output_dir,
            ".".join(
                [
                    sample.name,
                    f"tumour,immune,PDL1_image_overlay.ROI_{roi.roi_number}.svg",
                ]
            ),
        )

        # if not os.path.exists(f):
        fig = plot_overlaied_channels(roi, marker_combs)
        fig.savefig(f, **FIG_KWS)

        for i, region in enumerate(regions[roi.roi_number]):
            fig = plot_overlaied_channels(roi, marker_combs, region)
            fig.set_tight_layout(True)
            fig.savefig(
                pjoin(
                    output_dir,
                    ".".join(
                        [
                            sample.name,
                            f"example_reduced.tumour,immune,PDL1_image_overlay.ROI_{roi.roi_number}.example_{i + 1}.tight.svg",
                        ]
                    ),
                ),
                **FIG_KWS,
            )


def mcd():
    import imctools.io.mcdparser

    # have a look at the MCD file

    sample_name = "20200122"
    mcd_file = pjoin(
        "data", sample_name, "20200122_PD_L1_100pc", "20200122_PD_L1_100pc.mcd"
    )
    mcd = imctools.io.mcdparser.McdParser(mcd_file)
    pd.DataFrame([mcd.get_acquisition_channels(x) for x in mcd.acquisition_ids])


if __name__ == "__main__" and "get_ipython" not in locals():
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(1)


# # To perhaps reuse later
# def plot_channel_overlays_and_positive_cells():
#     # merges = [('PDL1(Lu175)', 'CD45(Sm152)', 'PanKeratin(Dy164)'), ('PDL1(Lu175)', 'CD3(Er170)', 'CD45(Sm152)')]
#     # merges = [('PanKeratin(Dy164)', 'PDL1(Lu175)'), ('CD45(Sm152)', 'PDL1(Lu175)'), ('CD3(Er170)', 'PDL1(Lu175)'), ('CD68(Tb159)', 'PDL1(Lu175)')]
#     roi_thresh = expr_thresh.join(cell_roi_number).query(f"roi == {roi.roi_number}")
#     cmap = get_transparent_cmaps(1, "binary")[0]
#     # aut = get_transparent_cmaps(1, "autumn")[0]
#     cmaps = get_transparent_cmaps(len(marker_combs[0]), "Set1")
#     n, m = len(marker_combs[0]) + 1, len(marker_combs)
#     fig, axes = plt.subplots(n, m, figsize=(m * 4, n * 4), gridspec_kw=dict(hspace=0, wspace=0), sharex=True, sharey=True)
#     for j, sp_markers in enumerate(marker_combs):
#         for i, marker in enumerate(sp_markers):
#             axes[i, j].set_title(marker)
#             axes[i, j].imshow(minmax_scale(eq(roi._get_channel(marker)[1].squeeze())), cmap=cmaps[i])
#             # axes[i, 1].imshow(eq(roi._get_channel(marker)[1].squeeze()), cmap=cmap)
#             # get positive cells
#             # posc = roi_thresh.loc[expr_thresh[marker] == 1]['index'].astype(int)
#             # axes[i, 1].contour(np.isin(roi.cell_mask, posc), linewidths=0.8, cmap=cmaps[i])

#         m = [[sp_markers[0], sp_markers[2], sp_markers[1]]] if len(sp_markers) == 3 else [[sp_markers[0], "empty", sp_markers[1]]]
#         for k, channels in enumerate(m):
#             k += 1
#             axes[i + k, j].set_title(", ".join(channels))

#             # # Simply sum up the channels
#             # roi.plot_channels(channels, axes=axes[i + k])

#             # Use RGB channel for real mixture
#             rgb = list()
#             for marker in channels:
#                 if marker == "empty":
#                     rgb.append(np.zeros(roi.shape[1:]))
#                 else:
#                     rgb.append(minmax_scale(roi._get_channel(marker)[1].squeeze()))
#             rgb = np.moveaxis(np.asarray(rgb), 0, -1)
#             # rgb = np.array(rgb.copy(), order='F')
#             # rgb.resize(rgb.shape[:2] + (4, ))
#             # rgb[:, :, 3] = rgb.mean(2)
#             # rgb = eq(rgb)
#             # TODO: add alpha channel
#             axes[i + k, j].imshow(rgb)
#             # axes[i + k, 1].imshow(rgb, cmap=cmap)

#             # # overlay
#             # for k, marker in enumerate(channels):
#             #     axes[i + k, 0].imshow(eq(roi._get_channel(marker)[1].squeeze()), cmap=cmaps[k])
#             #     axes[i + k, 1].imshow(eq(roi._get_channel(marker)[1].squeeze()), cmap=cmap)

#             # mark cells from all channels
#             # posc = pd.concat([roi_thresh.loc[(expr_thresh[marker] == 1)]['index'].astype(int) for marker in markers])
#             # axes[i + k, 1].contour(np.isin(roi.cell_mask, posc), linewidths=0.8, cmap=cmap)
#     for ax in axes.flatten():
#         ax.set_xticklabels([])
#         ax.set_yticklabels([])
#         ax.axis("off")
#     fig.savefig(output_prefix + "tumour,immune,PDL1_image_overlay.ROI_{roi.roi_number}.svg", **FIG_KWS)

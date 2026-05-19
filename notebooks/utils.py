"""Shared notebook configuration and I/O utilities."""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import anndata
import pandas as pd
from typing import Optional, Union

REPO_ROOT = Path(__file__).resolve().parent.parent


def setup_publication_style(base_dir: str) -> None:
    """Configure matplotlib for publication-quality figures.

    Sets rcParams for:
    - TrueType font embedding (pdf.fonttype=42)
    - Arial font from fonts/arial/ARIAL.TTF
    - Small font sizes (5-6 pt) for multi-panel figures
    - Thin axis lines (0.4 pt)
    - High DPI (600 save, 300 display)
    """
    # Font embedding for PDF
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    # Font sizes
    plt.rcParams['xtick.labelsize'] = 5
    plt.rcParams['ytick.labelsize'] = 5
    plt.rcParams['axes.labelsize'] = 6
    plt.rcParams['axes.titlesize'] = 6
    plt.rcParams['legend.fontsize'] = 6

    # DPI settings
    plt.rcParams["savefig.dpi"] = 600
    plt.rcParams['figure.dpi'] = 300

    # Load Arial font
    font_path = os.path.join(base_dir, "fonts/arial/ARIAL.TTF")
    fm.fontManager.addfont(font_path)
    prop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = prop.get_name()

    # Axis styling
    plt.rcParams["axes.linewidth"] = 0.4
    plt.rcParams["xtick.major.width"] = 0.4
    plt.rcParams["ytick.major.width"] = 0.4
    plt.rcParams["xtick.major.size"] = 3
    plt.rcParams["ytick.major.size"] = 3
    plt.rcParams["legend.frameon"] = False


def save_figure(
    fig_path: str,
    force_recompute: bool = False,
    format: str = "pdf",
    show: bool = True,
    fig: Optional[plt.Figure] = None,
) -> None:
    """Save figure with guard to avoid overwriting existing files.

    Args:
        fig_path: Path to save the figure.
        force_recompute: If True, overwrite existing file.
        format: Output format (default: "pdf").
        show: If True, call plt.show() after saving.
        fig: Figure to save. If None, saves current figure.
    """
    if not os.path.exists(fig_path) or force_recompute:
        if fig is not None:
            fig.savefig(fig_path, format=format, bbox_inches='tight')
        else:
            plt.savefig(fig_path, format=format, bbox_inches='tight')
        print("Saved figure")
    else:
        print("Figure already exists")
    if show:
        plt.show()


def save_figure_data(
    data: Union[pd.DataFrame, anndata.AnnData],
    fig_data_path: str,
    force_recompute: bool = False,
    index: bool = False,
) -> None:
    """Save figure source data for reproducibility.

    Supports DataFrames (saved as TSV) and AnnData objects (saved as h5ad).

    Args:
        data: DataFrame or AnnData containing the figure data.
        fig_data_path: Path to save the file.
        force_recompute: If True, overwrite existing file.
        index: If True, include DataFrame index in output (DataFrames only).
    """
    if not os.path.exists(fig_data_path) or force_recompute:
        if isinstance(data, anndata.AnnData):
            data.write(fig_data_path)
        else:
            data.to_csv(fig_data_path, sep="\t", index=index)
        print("Saved figure data")
    else:
        print("Figure data already exists")

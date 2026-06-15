from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from omegaconf import DictConfig

FIG_W = 5.65  # = textwidth_in
FIG_H1 = 3.1  # una fila de subplots
FIG_H2 = 6.2  # dos filas
FIG_H3 = 9.3  # tres filas


def apply_paper_style(cfg: DictConfig | None = None) -> None:
    # defaults
    textwidth = 6.0
    savefig_dpi = 300
    if cfg is not None:
        try:
            textwidth = float(cfg.viz.get("textwidth_in", textwidth))
            savefig_dpi = int(cfg.viz.get("savefig_dpi", savefig_dpi))
        except Exception:
            pass

    mpl.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "axes.titleweight": "normal",
            "figure.titlesize": 9,
            "figure.titleweight": "normal",
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": savefig_dpi,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    try:
        plt.style.use("seaborn-whitegrid")
    except Exception:
        pass

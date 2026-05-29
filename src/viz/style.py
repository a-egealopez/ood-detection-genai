from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from omegaconf import DictConfig


def apply_paper_style(cfg: DictConfig | None = None) -> None:
    # sensible defaults
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
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": 150,
            "savefig.dpi": savefig_dpi,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )

    try:
        plt.style.use("seaborn-whitegrid")
    except Exception:
        pass

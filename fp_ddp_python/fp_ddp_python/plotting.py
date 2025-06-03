"""
A logger class thats keeps track of the iterates.
"""
# Import standard libraries
from __future__ import annotations  # <-still need this.
from typing import TYPE_CHECKING

from matplotlib import pyplot as plt
import matplotlib
import numpy as np

# Import self-written libraries
if TYPE_CHECKING:
    from .logger import Logger

def latexify():
    params = {#'backend': 'ps',
            #'text.latex.preamble': r"\usepackage{amsmath}",
            'axes.labelsize': 10,
            'axes.titlesize': 10,
            'legend.fontsize': 10,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'text.usetex': True,
            'font.family': 'serif'
    }

    matplotlib.rcParams.update(params)

class Plotter:
    """
    A plotting class
    """
    def __init__(self) -> None:
        pass

    def plot_convergence_rate(self, data: list, labels: list, color=None, linestyles=None, save_name: str=None):

        latexify()

        # %% Plot Contraction of convergence criterion
        max_len = 0
        for sol in data:
            max_len = max(max_len, len(sol))

        # plt.figure(figsize=(4.5, 2.9))
        plt.figure(figsize=(4.5, 2.1)) #choice we took
        # plt.figure(figsize=(3.5, 2.0))
        # plt.figure(figsize=(5.0, 2.5)) # too large
        for i in range(len(data)):
            iters = list(range(len(data[i])))
            if linestyles != None:
                plt.semilogy(iters, data[i], label=labels[i], linestyle=linestyles[i], color=color[i])
            else:
                plt.semilogy(iters, data[i], label=labels[i])
        plt.xlabel('iteration number')
        plt.ylabel('KKT residual')
        # plt.grid(alpha=0.5)
        plt.tight_layout()
        plt.legend()
        plt.xlim((0, max_len))
        plt.ylim((1e-8, 1e-1))

        if save_name is not None:
            plt.savefig(save_name + ".pdf",
                        dpi=300,
                        bbox_inches='tight',
                        pad_inches=0.01)
        plt.show()
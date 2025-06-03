import pickle
from matplotlib import pyplot as plt 
import numpy as np
import os

dir_name = os.path.dirname(os.path.abspath(__file__))

import matplotlib
def latexify():
    params = {#'backend': 'ps',
            #'text.latex.preamble': r"\usepackage{amsmath}",
            'axes.labelsize': 9.5,
            'axes.titlesize': 10,
            'legend.fontsize': 10,
            'xtick.labelsize': 10,
            'ytick.labelsize': 9.5,
            'text.usetex': True,
            'font.family': 'serif'
    }

    matplotlib.rcParams.update(params)

latexify()

###############################################################################
# DDP
###############################################################################
# Open files
new_ddp_file_timings = open(dir_name + '/acados_ddp_results/new_acados_n_eval_gn_hessian.pkl', 'rb')
ddp_file_n_eval_gn_hessian = open(dir_name + '/acados_ddp_results/acados_n_eval_gn_hessian.pkl', 'rb')
# Load stuff
new_ddp_n_eval_gn_hessian = pickle.load(new_ddp_file_timings)
ddp_n_eval_gn_hessian = pickle.load(ddp_file_n_eval_gn_hessian)
# Close files
new_ddp_file_timings.close()
ddp_file_n_eval_gn_hessian.close()

print("new DDp: ",new_ddp_n_eval_gn_hessian)
print("old DDp: ",ddp_n_eval_gn_hessian)
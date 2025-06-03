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
ddp_file_timings = open(dir_name + '/acados_ddp_results/acados_timings.pkl', 'rb')
ddp_file_n_eval_gn_hessian = open(dir_name + '/ddp_results/ddp_n_eval_gn_hessian.pkl', 'rb')
# Load stuff
ddp_timings = pickle.load(ddp_file_timings)
print('acados ddp timings: ', ddp_timings)
ddp_n_eval_gn_hessian = pickle.load(ddp_file_n_eval_gn_hessian)
# Close files
ddp_file_timings.close()
ddp_file_n_eval_gn_hessian.close()

###############################################################################
# SQP
###############################################################################
# Open files
sqp_file_timings = open(dir_name + '/acados_ddp_results/acados_sqp_timings.pkl', 'rb')
sqp_file_n_eval_gn_hessian = open(dir_name + '/acados_ddp_results/acados_sqp_n_eval_gn_hessian.pkl', 'rb')
# Load stuff
sqp_timings = pickle.load(sqp_file_timings)
print('acados sqp timings: ', sqp_timings)
sqp_n_eval_gn_hessian = pickle.load(sqp_file_n_eval_gn_hessian)
# Close files
sqp_file_timings.close()
sqp_file_n_eval_gn_hessian.close()

###############################################################################
# SCIPY
###############################################################################
# Open files
scipy_file_n_eval_hessian_f = open(dir_name + '/scipy_newton_cg_results/scipy_n_eval_hess_f.pkl', 'rb')
# Load stuff
scipy_n_eval_hessian_f = pickle.load(scipy_file_n_eval_hessian_f)
print('scipy n_eval_hessian_f: ', scipy_n_eval_hessian_f)
# Close files
scipy_file_n_eval_hessian_f.close()

###############################################################################
# IPOPT
###############################################################################
# Open files
ipopt_file_n_eval_hessian_l = open(dir_name + '/ipopt_no_codegen/ipopt_n_eval_hessian_l.pkl', 'rb')
ipopt_codegen_file_timings = open(dir_name + '/ipopt_code_generation/ipopt_codegen_timings.pkl', 'rb')
ipopt_file_timings = open(dir_name + '/ipopt_no_codegen/ipopt_t_wall_total.pkl', 'rb')
# Load stuff
ipopt_timings = pickle.load(ipopt_file_timings)
ipopt_codegen_timings = pickle.load(ipopt_codegen_file_timings)
print('ipopt codegen timings: ', ipopt_codegen_timings)
ipopt_n_eval_hessian_l = pickle.load(ipopt_file_n_eval_hessian_l)
# Close files
ipopt_file_timings.close()
ipopt_codegen_file_timings.close()
ipopt_file_n_eval_hessian_l.close()

###############################################################################
# FATROP
###############################################################################
# Open files
fatrop_file_n_eval_hessian_l = open(dir_name + '/fatrop_results/fatrop_n_eval_hessian_l.pkl', 'rb')
fatrop_file_timings = open(dir_name + '/fatrop_results/fatrop_t_wall_total.pkl', 'rb')
# Load stuff
fatrop_timings = pickle.load(fatrop_file_timings)
print('fatrop timings: ', fatrop_timings)
fatrop_n_eval_hessian_l = pickle.load(fatrop_file_n_eval_hessian_l)
print('fatrop_n_eval_hessian_l: ', fatrop_n_eval_hessian_l)
# Close files
fatrop_file_timings.close()
fatrop_file_n_eval_hessian_l.close()

# Plot objective evaluations
x_pos = np.linspace(0.7, 4.3, 100)

# Polish results for plotting
for i in range(100):
    if ddp_timings[i] == -1 :
        ddp_timings[i] = np.inf
    if ipopt_n_eval_hessian_l[i] == -1 or ipopt_n_eval_hessian_l[i] == np.inf:
        ipopt_n_eval_hessian_l[i] = np.inf
        ipopt_codegen_timings[i] = np.inf
        ipopt_timings[i] = np.inf
    if scipy_n_eval_hessian_f[i] == -1:
        scipy_n_eval_hessian_f[i] = np.inf
    if fatrop_n_eval_hessian_l[i] == -1 or fatrop_n_eval_hessian_l[i] == np.inf:
        fatrop_n_eval_hessian_l[i] = np.inf
        fatrop_timings[i] = np.inf

###############################################################################
# Plotting from here
###############################################################################
title = "wall_time_and_hessian"
fig, (ax1, ax2) = plt.subplots(1, 2, sharey = True, figsize = (5.0, 2.0))

# This is the original figsize from the final version
plt.subplots_adjust(wspace=0.05)

labels = ["FP-DDP", "$\\texttt{IPOPT}$", "$\\texttt{FATROP}$"]
markers = ['o', 'v', '^']
colors = ['tab:blue', 'tab:green', 'tab:orange']
lstyle = ["solid", "dashed", "dotted"]
# Plot performance plot time
# Do performance plot here for Hessian evaluations
t_sp = np.vstack((np.vstack((np.array(ddp_timings), np.array(ipopt_codegen_timings))), np.array(fatrop_timings)))
n_p = t_sp.shape[1] # number of problems
n_s = t_sp.shape[0] # number of solvers
tm_sp = np.min(t_sp, axis=0) # take minimum of solvers per problem
tm_sp = np.tile(tm_sp, (n_s, 1)) # duplicate the minima for number of solvers solvers
r_sp = t_sp # do not divide all entries of the metric by the minimum (stay relative)
τ_max = np.max(r_sp[np.isfinite(r_sp)]) # get the maximum tau that is necessary
# for ρ_solver, label, linestyle in zip(r_sp, labels, lstyle):
for ρ_solver, label, color, linestyle in zip(r_sp, labels, colors, lstyle):
    ρ_solver = ρ_solver[np.isfinite(ρ_solver)] #remove all values with inf
    x, y = np.unique(ρ_solver, return_counts=True) #get unique values and count them
    y = (1. / n_p) * np.cumsum(y) * 100 # get fraction of solved problems
    x = np.append(x, [τ_max])
    y = np.append(y, [y[-1]])
    # ax1.step(x, y, where='post', label=label, color=color)
    ax1.step(x, y, where='post', label=label, linestyle=linestyle, color='k')

ax1.set_xscale("log")
ax1.set_xlim(0, τ_max*0.99)
ax1.set_ylim(0, 102)
ax1.set_xlabel('(a) time $t$ [s]')
ax1.set_ylabel('$\\#$ solved problems')
ax1.grid()

# Plot performance plot Hessian
# Do performance plot here for Hessian evaluations
t_sp = np.vstack((np.vstack((np.vstack((np.array(ddp_n_eval_gn_hessian), np.array(ipopt_n_eval_hessian_l))), np.array(scipy_n_eval_hessian_f))), np.array(fatrop_n_eval_hessian_l)))
n_p = t_sp.shape[1]
n_s = t_sp.shape[0]
tm_sp = np.min(t_sp, axis=0)
tm_sp = np.tile(tm_sp, (n_s, 1))
r_sp = t_sp / tm_sp

labels = ["FP-DDP", "$\\texttt{IPOPT}$", "$\\texttt{Scipy}$", "$\\texttt{FATROP}$"]
colors = ['tab:blue', 'tab:green', 'tab:orange', 'tab:purple']
lstyle = ["solid", "dashed", "dashdot", "dotted"]

tau_max = np.max(r_sp[np.isfinite(r_sp)]) * 2
for ρ_solver, label, linestyle in zip(r_sp, labels, lstyle):
# for ρ_solver, label, color in zip(r_sp, labels, colors):
    ρ_solver = ρ_solver[np.isfinite(ρ_solver)]
    x, y = np.unique(ρ_solver, return_counts=True)
    y = (1. / n_p) * np.cumsum(y) * 100
    x = np.append(x, [tau_max])
    y = np.append(y, [y[-1]])
    ax2.step(x, y, where='post', label=label, linestyle=linestyle, color='k')
    # ax2.step(x, y, where='post', label=label, color=color)

ax2.set_xscale("log", base=2) #if other base is used there is not much to see
ax2.set_xlim(1, tau_max * 0.99)
ax2.set_ylim(0, 102)
ax2.grid()
ax2.set_xlabel('(b) Hessian evaluation ratio')
# So far, nothing special except the managed prop_cycle. Now the trick:
lines_labels = [ax.get_legend_handles_labels() for ax in plt.gcf().axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
print(labels)
print(lines)
lines.pop(3)
lines.pop(3)
lines.pop(-1)
labels.pop(3)
labels.pop(3)
labels.pop(-1)
# Finally, the legend (that maybe you'll customize differently)
ax2.legend(lines, labels, handlelength=1.3)
plt.savefig("phd_thesis_performance_plot_"+title+"_cart_pendulum.pdf", dpi=300, bbox_inches='tight', pad_inches=0.01)
plt.tight_layout()
plt.show()

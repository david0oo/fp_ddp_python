import pickle
from matplotlib import pyplot as plt 
import numpy as np
import os

dir_name = os.path.dirname(os.path.abspath(__file__))

import matplotlib
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

latexify()
moon_x = np.linspace(0.7, 4.3, 100)

###############################################################################
# DDP
###############################################################################
# Open files
ddp_file_n_eval_f = open(dir_name + '/ddp_results/ddp_n_eval_f.pkl', 'rb')
ddp_file_n_eval_g = open(dir_name + '/ddp_results/ddp_n_eval_g.pkl', 'rb')
ddp_file_n_eval_gradient_f = open(dir_name + '/ddp_results/ddp_n_eval_gradient_f.pkl', 'rb')
ddp_file_n_eval_jac_g = open(dir_name + '/ddp_results/ddp_n_eval_jacobian_g.pkl', 'rb')
ddp_file_n_eval_gn_hessian = open(dir_name + '/ddp_results/ddp_n_eval_gn_hessian.pkl', 'rb')

# Load stuff
ddp_n_eval_f = pickle.load(ddp_file_n_eval_f)
ddp_n_eval_g = pickle.load(ddp_file_n_eval_g)
ddp_n_eval_gradient_f = pickle.load(ddp_file_n_eval_gradient_f)
ddp_n_eval_jacobian_g = pickle.load(ddp_file_n_eval_jac_g)
ddp_n_eval_gn_hessian = pickle.load(ddp_file_n_eval_gn_hessian)

# Close files
ddp_file_n_eval_f.close()
ddp_file_n_eval_g.close()
ddp_file_n_eval_gradient_f.close()
ddp_file_n_eval_jac_g.close()
ddp_file_n_eval_gn_hessian.close()

###############################################################################
# SCIPY
###############################################################################
# Open files
scipy_file_n_eval_f = open(dir_name + '/scipy_newton_cg_results/scipy_n_eval_f.pkl', 'rb')
scipy_file_n_eval_gradient_f = open(dir_name + '/scipy_newton_cg_results/scipy_n_eval_gradient_f.pkl', 'rb')
scipy_file_n_eval_hessian_f = open(dir_name + '/scipy_newton_cg_results/scipy_n_eval_hess_f.pkl', 'rb')

# Load stuff
scipy_n_eval_f = pickle.load(scipy_file_n_eval_f)
scipy_n_eval_gradient_f = pickle.load(scipy_file_n_eval_gradient_f)
scipy_n_eval_hessian_f = pickle.load(scipy_file_n_eval_hessian_f)

# Close files
scipy_file_n_eval_f.close()
scipy_file_n_eval_gradient_f.close()
scipy_file_n_eval_hessian_f.close()

###############################################################################
# IPOPT
###############################################################################
# Open files
ipopt_file_n_eval_f = open(dir_name + '/ipopt_results/ipopt_n_eval_f.pkl', 'rb')
ipopt_file_n_eval_g = open(dir_name + '/ipopt_results/ipopt_n_eval_g.pkl', 'rb')
ipopt_file_n_eval_gradient_f = open(dir_name + '/ipopt_results/ipopt_n_eval_gradient_f.pkl', 'rb')
ipopt_file_n_eval_jacobian_g = open(dir_name + '/ipopt_results/ipopt_n_eval_jacobian_g.pkl', 'rb')
ipopt_file_n_eval_hessian_l = open(dir_name + '/ipopt_results/ipopt_n_eval_hessian_l.pkl', 'rb')

# Load stuff
ipopt_n_eval_f = pickle.load(ipopt_file_n_eval_f)
ipopt_n_eval_g = pickle.load(ipopt_file_n_eval_g)
ipopt_n_eval_gradient_f = pickle.load(ipopt_file_n_eval_gradient_f)
ipopt_n_eval_jacobian_g = pickle.load(ipopt_file_n_eval_jacobian_g)
ipopt_n_eval_hessian_l = pickle.load(ipopt_file_n_eval_hessian_l)

# Close files
ipopt_file_n_eval_f.close()
ipopt_file_n_eval_g.close()
ipopt_file_n_eval_gradient_f.close()
ipopt_file_n_eval_jacobian_g.close()
ipopt_file_n_eval_hessian_l.close()

# Plot objective evaluations
x_pos = np.linspace(0.7, 4.3, 100)

for i in range(100):

    if ipopt_n_eval_f[i] == -1:
        ipopt_n_eval_f[i] = np.inf
    if ddp_n_eval_f[i] == -1:
        ddp_n_eval_f[i] = np.inf
    if scipy_n_eval_f[i] == -1:
        scipy_n_eval_f[i] = np.inf

    if ipopt_n_eval_gradient_f[i] == -1:
        ipopt_n_eval_gradient_f[i] = np.inf
    if ddp_n_eval_gradient_f[i] == -1:
        ddp_n_eval_gradient_f[i] = np.inf
    if scipy_n_eval_gradient_f[i] == -1:
        scipy_n_eval_gradient_f[i] = np.inf

    if ipopt_n_eval_hessian_l[i] == -1:
        ipopt_n_eval_hessian_l[i] = np.inf
    if ddp_n_eval_gn_hessian[i] == -1:
        ddp_n_eval_gn_hessian[i] = np.inf
    if scipy_n_eval_hessian_f[i] == -1:
        scipy_n_eval_hessian_f[i] = np.inf


# Plot performance plot objective
# Do performance plot here for Hessian evaluations
t_sp = np.vstack((np.vstack((np.array(ddp_n_eval_gradient_f), np.array(ipopt_n_eval_gradient_f))), np.array(scipy_n_eval_gradient_f)))
n_p = t_sp.shape[1]
n_s = t_sp.shape[0]
tm_sp = np.min(t_sp, axis=0)
tm_sp = np.tile(tm_sp, (n_s, 1))
r_sp = t_sp / tm_sp

title = "Objective_Function_Evaluations"
plt.figure()

labels = ["FPP-DDP", "IPOPT-EXACT", "Scipy"]

tau_max = np.max(r_sp[np.isfinite(r_sp)]) * 2
for ρ_solver, label in zip(r_sp, labels):
    ρ_solver = ρ_solver[np.isfinite(ρ_solver)]
    x, y = np.unique(ρ_solver, return_counts=True)
    y = (1. / n_p) * np.cumsum(y)
    x = np.append(x, [tau_max])
    y = np.append(y, [y[-1]])
    plt.step(x, y, '-', where='post', label=label)

plt.xscale("log", base=2)
plt.xlim(1, tau_max * 0.99)
plt.xlabel(r'$\tau$')
plt.ylabel(r'$\rho_s(\tau)$')
# plt.ylim(0, None)
plt.legend(loc='lower right')
plt.savefig(title+".pdf", dpi=300, bbox_inches='tight', pad_inches=0.01)
plt.tight_layout()
plt.show()

# Plot performance plot Gradient
# Do performance plot here for Hessian evaluations
t_sp = np.vstack((np.vstack((np.array(ddp_n_eval_gradient_f), np.array(ipopt_n_eval_gradient_f))), np.array(scipy_n_eval_gradient_f)))

n_p = t_sp.shape[1]
n_s = t_sp.shape[0]
tm_sp = np.min(t_sp, axis=0)
tm_sp = np.tile(tm_sp, (n_s, 1))
r_sp = t_sp / tm_sp

title = "Gradient_Evaluations"
plt.figure()

tau_max = np.max(r_sp[np.isfinite(r_sp)]) * 2
for ρ_solver, label in zip(r_sp, labels):
    ρ_solver = ρ_solver[np.isfinite(ρ_solver)]
    x, y = np.unique(ρ_solver, return_counts=True)
    y = (1. / n_p) * np.cumsum(y)
    x = np.append(x, [tau_max])
    y = np.append(y, [y[-1]])
    plt.step(x, y, '-', where='post', label=label)

plt.xscale("log", base=2)
plt.xlim(1, tau_max * 0.99)
plt.xlabel(r'$\tau$')
plt.ylabel(r'$\rho_s(\tau)$')
# plt.ylim(0, None)
plt.legend(loc='lower right')
plt.savefig(title+".pdf", dpi=300, bbox_inches='tight', pad_inches=0.01)
plt.tight_layout()
plt.show()

# Plot performance plot Hessian
# Do performance plot here for Hessian evaluations
t_sp = np.vstack((np.vstack((np.array(ddp_n_eval_gn_hessian), np.array(ipopt_n_eval_hessian_l))), np.array(scipy_n_eval_hessian_f)))
n_p = t_sp.shape[1]
n_s = t_sp.shape[0]
tm_sp = np.min(t_sp, axis=0)
tm_sp = np.tile(tm_sp, (n_s, 1))
r_sp = t_sp / tm_sp

title = "hessian_evaluations"
plt.figure(figsize=(4.5, 2.0))

labels = ["FP-DDP", "$\\texttt{IPOPT}$", "$\\texttt{Scipy}$"]
lstyle = ["solid", "dashed", "dotted"]

tau_max = np.max(r_sp[np.isfinite(r_sp)]) * 2
for ρ_solver, label, linestyle in zip(r_sp, labels, lstyle):
    ρ_solver = ρ_solver[np.isfinite(ρ_solver)]
    x, y = np.unique(ρ_solver, return_counts=True)
    y = (1. / n_p) * np.cumsum(y)
    x = np.append(x, [tau_max])
    y = np.append(y, [y[-1]])
    plt.step(x, y, '-', where='post', label=label, linestyle=linestyle, color='k')

plt.xscale("log", base=2)
plt.xlim(1, tau_max * 0.99)
plt.xlabel(r'$\tau$')
plt.ylabel(r'$\rho_s(\tau)$')
plt.legend(loc='lower right')
plt.savefig("performance_plot"+title+"_cart_pendulum.pdf", dpi=300, bbox_inches='tight', pad_inches=0.01)
plt.tight_layout()
plt.show()
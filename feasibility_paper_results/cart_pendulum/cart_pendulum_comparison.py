# Include standard 
import casadi as cs
import rockit as roc
import numpy as np
import sys
import os
import scipy.optimize as opt
print(os.path.abspath(__file__))
dir_name = os.path.dirname(os.path.abspath(__file__))
import pickle
include_dir = os.path.join(dir_name, "..", "..","test_problems")
sys.path.append(include_dir)

from cartpendulum_time_optimal_obstacle import create_ocp
from fp_ddp_python.solver import FeasibilityProblemSolver
from fp_ddp_python.ocp_transformer import OCP_To_Data_Transformer, OCP_To_Unconstrained_NLP_Transformer
from fp_ddp_python.plotting import Plotter


###############################################################################
# DDP Solver 
###############################################################################

def test_ddp(ocp: roc.Ocp, plot=False):
    transformer = OCP_To_Data_Transformer()
    feasibility_problem_data = transformer.transform(ocp,smoothmax=False)

    solver = FeasibilityProblemSolver(feasibility_problem_data, mode="ddp", init_feasible=True)
    fail = False
    solver.solve(feasibility_problem_data)

    if plot:
        plotter = Plotter()
        plotter.plot_convergence_rate([solver.log.stationarity], ["Stationarity"])
        plotter.plot_convergence_rate([solver.log.nlp_infeasibility], ["Original Infeasibility"])
        plotter.plot_convergence_rate([solver.log.kkt_residual], ["KKT residual"])

    if solver.iterate.f_k > 1e-8 or fail:
        print("return -1...")
        return (-1, -1, -1, -1, -1)
    else:
        return (solver.log.n_eval_f, solver.log.n_eval_g, 
                solver.log.n_eval_gradient_f, solver.log.n_eval_jacobian_g,
                solver.log.n_eval_hessian_lagrangian)

###############################################################################
# Scipy CG solver
###############################################################################

def test_scipy(ocp: roc.Ocp):
    transformer = OCP_To_Unconstrained_NLP_Transformer()
    feasibility_problem_data = transformer.transform(ocp,smoothmax=False)

    x0 = feasibility_problem_data['initial_point']
    p0 = feasibility_problem_data['parameter_values']
    f_fun = feasibility_problem_data['feas_nlp_f_fun']
    grad_f_fun = feasibility_problem_data['feas_nlp_grad_f_fun']
    hessian_f_fun = feasibility_problem_data['feas_nlp_hessian_fun']

    def f(x:np.array):
        return np.array(f_fun(x, p0))

    def grad_f(x:np.array):
        return np.array(grad_f_fun(x, p0)).squeeze()

    def hessian_f(x:np.array):
        return np.array(hessian_f_fun(x, p0))

    result = opt.minimize(f, np.array(x0).squeeze(), method="Newton-CG", jac=grad_f, hess=hessian_f, tol=1e-8)
    
    print("Objective value", result.fun)
    print("Number of iterations:", result.nit)
    print("Number of function evaluations:", result.nfev)
    print("Number of gradient evaluations:", result.njev)
    print("Number of gradient evaluations:", result.nhev)
    if result.fun > 1e-8:
        return -1, -1, -1, -1
    else:
        return result.fun, result.nfev, result.njev, result.nhev
    

###############################################################################
# IPOPT
###############################################################################
def test_ipopt_feasibility_problem(ocp: roc.Ocp):

    transformer = OCP_To_Data_Transformer()
    feasibility_problem_data = transformer.transform(ocp,smoothmax=False)

    obj = feasibility_problem_data['feas_nlp_f']
    g = feasibility_problem_data['feas_nlp_g']
    x = ocp._method.opti.x
    p = ocp._method.opti.p
    lbg = feasibility_problem_data['feas_nlp_lbg']
    ubg = feasibility_problem_data['feas_nlp_ubg']

    x0_vector = ocp.initial_value(ocp._method.opti.x)
    p0_vector = ocp.initial_value(ocp._method.opti.p)

    nlp = {'x':x, 'p':p, 'f':obj, 'g':g}
    solver = cs.nlpsol('solver', 'ipopt', nlp)
    res = solver(x0=x0_vector, p=p0_vector, lbg=lbg, ubg=ubg)

    stats = solver.stats()
    if res['f'] > 1e-8 or not stats['success']:
        return (np.inf, np.inf, np.inf, np.inf, np.inf)
    else:
        return (stats['n_call_nlp_f'], stats['n_call_nlp_g'], 
            stats['n_call_nlp_grad_f'], stats['n_call_nlp_jac_g'],
            stats['n_call_nlp_hess_l'])

###############################################################################
# Test the problems
###############################################################################
def test_on_cart_pendulum(ddp=True, scipy=False, ipopt=False):
    
    # Prepare lists for data
    moon_x = cs.linspace(0.7, 4.3, 100)
    ddp_list_n_eval_f = []
    ddp_list_n_eval_g = []
    ddp_list_n_eval_gradient_f = []
    ddp_list_n_eval_jacobian_g = []
    ddp_list_n_eval_gn_hessian = []

    scipy_list_f_values = []
    scipy_list_n_eval_f = []
    scipy_list_n_eval_gradient_f = []
    scipy_list_n_eval_hess_f = []

    ipopt_list_n_eval_f = []
    ipopt_list_n_eval_g = []
    ipopt_list_n_eval_gradient_f = []
    ipopt_list_n_eval_jacobian_g = []
    ipopt_list_n_eval_hessian_l = []
    ipopt_fails = []
    # Run the simulation
    for i in range(100):

        ocp = create_ocp(moon_x=moon_x[i])

        if ddp:
            n_eval_f, n_eval_g, n_eval_gradient_f, n_eval_jacobian_g, n_eval_gn_hessian = test_ddp(ocp)
            ddp_list_n_eval_f.append(n_eval_f)
            ddp_list_n_eval_g.append(n_eval_g)
            ddp_list_n_eval_gradient_f.append(n_eval_gradient_f)
            ddp_list_n_eval_jacobian_g.append(n_eval_jacobian_g)
            ddp_list_n_eval_gn_hessian.append(n_eval_gn_hessian)
        elif scipy:
            obj_value, n_eval_f, n_eval_gradient_f, n_eval_hess_f = test_scipy(ocp)
            scipy_list_f_values.append(obj_value)
            scipy_list_n_eval_f.append(n_eval_f)
            scipy_list_n_eval_gradient_f.append(n_eval_gradient_f)
            scipy_list_n_eval_hess_f.append(n_eval_hess_f)
        elif ipopt:
            n_eval_f, n_eval_g, n_eval_gradient_f, n_eval_jac_g, n_eval_hess_l = test_ipopt_feasibility_problem(ocp)
            ipopt_list_n_eval_f.append(n_eval_f)
            ipopt_list_n_eval_g.append(n_eval_g)
            ipopt_list_n_eval_gradient_f.append(n_eval_gradient_f)
            ipopt_list_n_eval_jacobian_g.append(n_eval_jac_g)
            ipopt_list_n_eval_hessian_l.append(n_eval_hess_l)
            if n_eval_hess_l == np.inf:
                ipopt_fails.append(i)


    # Store the program
    if ddp:
        with open('ddp_results/ddp_n_eval_f.pkl', 'wb') as f:
            pickle.dump(ddp_list_n_eval_f, f)
        with open('ddp_results/ddp_n_eval_g.pkl', 'wb') as f:
            pickle.dump(ddp_list_n_eval_g, f)
        with open('ddp_results/ddp_n_eval_gradient_f.pkl', 'wb') as f:
            pickle.dump(ddp_list_n_eval_gradient_f, f)
        with open('ddp_results/ddp_n_eval_jacobian_g.pkl', 'wb') as f:
            pickle.dump(ddp_list_n_eval_jacobian_g, f)
        with open('ddp_results/ddp_n_eval_gn_hessian.pkl', 'wb') as f:
            pickle.dump(ddp_list_n_eval_gn_hessian, f)

    if scipy:
        with open('scipy_newton_cg_results/scipy_f_values.pkl', 'wb') as f:
            pickle.dump(scipy_list_f_values, f)
        with open('scipy_newton_cg_results/scipy_n_eval_f.pkl', 'wb') as f:
            pickle.dump(scipy_list_n_eval_f, f)
        with open('scipy_newton_cg_results/scipy_n_eval_gradient_f.pkl', 'wb') as f:
            pickle.dump(scipy_list_n_eval_gradient_f, f)
        with open('scipy_newton_cg_results/scipy_n_eval_hess_f.pkl', 'wb') as f:
            pickle.dump(scipy_list_n_eval_hess_f, f)

    if ipopt:
        with open('ipopt_results/ipopt_n_eval_f.pkl', 'wb') as f:
            pickle.dump(ipopt_list_n_eval_f, f)
        with open('ipopt_results/ipopt_n_eval_g.pkl', 'wb') as f:
            pickle.dump(ipopt_list_n_eval_g, f)
        with open('ipopt_results/ipopt_n_eval_gradient_f.pkl', 'wb') as f:
            pickle.dump(ipopt_list_n_eval_gradient_f, f)
        with open('ipopt_results/ipopt_n_eval_jacobian_g.pkl', 'wb') as f:
            pickle.dump(ipopt_list_n_eval_jacobian_g, f)
        with open('ipopt_results/ipopt_n_eval_hessian_l.pkl', 'wb') as f:
            pickle.dump(ipopt_list_n_eval_hessian_l, f)

###############################################################################
# Run the simulation
###############################################################################
if __name__ == "__main__":
    test_on_cart_pendulum(ddp=False, scipy=False, ipopt=True)

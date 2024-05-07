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
include_dir = os.path.join(dir_name, "..", "..", "..","test_problems")
sys.path.append(include_dir)

from parametric_cartpendulum_time_optimal_obstacle import create_ocp
from fp_ddp_python.solver import FeasibilityProblemSolver
from fp_ddp_python.ocp_transformer import OCP_To_Data_Transformer, OCP_To_Unconstrained_NLP_Transformer
from fp_ddp_python.plotting import Plotter

###############################################################################
# Test the problems
###############################################################################
def test_on_cart_pendulum():

    n_tests = 1#20

    ipopt_list_n_eval_f = []
    ipopt_list_n_eval_g = []
    ipopt_list_n_eval_gradient_f = []
    ipopt_list_n_eval_jacobian_g = []
    ipopt_list_n_eval_hessian_l = []
    ipopt_list_t_wall_total = []

    ocp = create_ocp()

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

    moon_x = cs.linspace(0.7, 4.3, 100)
    ipopt_fails = []

    # Run the simulation
    n_call_f = 0
    n_call_g = 0
    n_call_grad_f = 0
    n_call_jac_g = 0
    n_call_hess_l = 0
    t_wall_tmp = 0.0
    for i in range(100):

        t_tmp = []
        for j in range(n_tests):
            res = solver(x0=x0_vector, p=np.array([moon_x[i]]), lbg=lbg, ubg=ubg)

            stats = solver.stats()
            print("Stats are: ", stats['iterations']['inf_pr'][-1])
            # if res['f'] > 1e-8 or not stats['success']:
            if res['f'] > 1e-8 or stats['iterations']['inf_pr'][-1] > 1e-8:
                t_tmp.append(np.inf)
                n_eval_f = np.inf
                n_eval_g = np.inf
                n_eval_gradient_f = np.inf
                n_eval_jac_g = np.inf
                n_eval_hess_l = np.inf
                ipopt_fails.append(i)
            else:
                t_tmp.append(stats['t_wall_total'])
                n_eval_f = stats['n_call_nlp_f'] - n_call_f
                n_call_f = stats['n_call_nlp_f']
                n_eval_g = stats['n_call_nlp_g'] - n_call_g
                n_call_g = stats['n_call_nlp_g']
                n_eval_gradient_f = stats['n_call_nlp_grad_f'] - n_call_grad_f
                n_call_grad_f = stats['n_call_nlp_grad_f']
                n_eval_jac_g = stats['n_call_nlp_jac_g'] - n_call_jac_g
                n_call_jac_g = stats['n_call_nlp_jac_g']
                n_eval_hess_l = stats['n_call_nlp_hess_l'] - n_call_hess_l
                n_call_hess_l = stats['n_call_nlp_hess_l']
        ipopt_list_t_wall_total.append(min(t_tmp))
        ipopt_list_n_eval_f.append(n_eval_f)
        ipopt_list_n_eval_g.append(n_eval_g)
        ipopt_list_n_eval_gradient_f.append(n_eval_gradient_f)
        ipopt_list_n_eval_jacobian_g.append(n_eval_jac_g)
        ipopt_list_n_eval_hessian_l.append(n_eval_hess_l)

    print("t_wall total: ", ipopt_list_t_wall_total)
    print("IPOPT fails: ", ipopt_fails)

    with open('ipopt_t_wall_total.pkl', 'wb') as f:
        pickle.dump(ipopt_list_n_eval_hessian_l, f)
    with open('ipopt_n_eval_f.pkl', 'wb') as f:
        pickle.dump(ipopt_list_n_eval_f, f)
    with open('ipopt_n_eval_g.pkl', 'wb') as f:
        pickle.dump(ipopt_list_n_eval_g, f)
    with open('ipopt_n_eval_gradient_f.pkl', 'wb') as f:
        pickle.dump(ipopt_list_n_eval_gradient_f, f)
    with open('ipopt_n_eval_jacobian_g.pkl', 'wb') as f:
        pickle.dump(ipopt_list_n_eval_jacobian_g, f)
    with open('ipopt_n_eval_hessian_l.pkl', 'wb') as f:
        pickle.dump(ipopt_list_n_eval_hessian_l, f)

###############################################################################
# Run the simulation
###############################################################################
if __name__ == "__main__":
    test_on_cart_pendulum()

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
def test_on_cart_pendulum(solve_feasibility_problem=False):

    n_tests = 20

    fatrop_list_n_eval_f = []
    fatrop_list_n_eval_g = []
    fatrop_list_n_eval_gradient_f = []
    fatrop_list_n_eval_jacobian_g = []
    fatrop_list_n_eval_hessian_l = []
    fatrop_list_t_wall_total = []

    ocp = create_ocp()

    transformer = OCP_To_Data_Transformer()
    feasibility_problem_data = transformer.transform(ocp,smoothmax=False)

    if solve_feasibility_problem:
        obj = feasibility_problem_data['feas_nlp_f']
        g = feasibility_problem_data['feas_nlp_g']
        x = ocp._method.opti.x
        p = ocp._method.opti.p
        lbg = feasibility_problem_data['feas_nlp_lbg']
        ubg = feasibility_problem_data['feas_nlp_ubg']
    else:
        obj = 0
        g = feasibility_problem_data['nlp_g']
        x = ocp._method.opti.x
        p = ocp._method.opti.p
        lbg = feasibility_problem_data['nlp_lbg']
        ubg = feasibility_problem_data['nlp_ubg']

    x0_vector = ocp.initial_value(ocp._method.opti.x)
    p0_vector = ocp.initial_value(ocp._method.opti.p)

    nlp = {'x':x, 'p':p, 'f':obj, 'g':g}
    N = 100
    nx = 5
    nu = 1
    solver = cs.nlpsol('solver', 'fatrop', nlp, {'expand':True, 'structure_detection':'auto',
                                                 'debug':False, 'equality': N*nx*[True],
                                                # 'cache':{'nlp_hess_l':feasibility_problem_data['feas_nlp_casadi_gn_hessian']},
                                                 'jit':True, 'jit_temp_suffix':False, 'jit_options':{'flags': ['-O3'], 'compiler': 'gcc'}})

    # solver = cs.nlpsol('solver', 'fatrop', nlp)
    moon_x = cs.linspace(0.7, 4.3, 100)
    fatrop_fails = []

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
            if res['f'] > 1e-8:
                t_tmp.append(np.inf)
                n_eval_f = np.inf
                n_eval_g = np.inf
                n_eval_gradient_f = np.inf
                n_eval_jac_g = np.inf
                n_eval_hess_l = np.inf
                fatrop_fails.append(i)
            else:
                t_tmp.append(stats['fatrop']['time_total'])
                n_eval_f = stats['fatrop']['eval_obj_count']
                n_eval_g = stats['fatrop']['eval_cv_count']
                n_eval_gradient_f = stats['fatrop']['eval_grad_count']
                n_eval_jac_g = stats['fatrop']['eval_jac_count']
                n_eval_hess_l = stats['fatrop']['eval_hess_count']
        fatrop_list_t_wall_total.append(min(t_tmp))
        fatrop_list_n_eval_f.append(n_eval_f)
        fatrop_list_n_eval_g.append(n_eval_g)
        fatrop_list_n_eval_gradient_f.append(n_eval_gradient_f)
        fatrop_list_n_eval_jacobian_g.append(n_eval_jac_g)
        fatrop_list_n_eval_hessian_l.append(n_eval_hess_l)

    print("t_wall total: ", fatrop_list_t_wall_total)
    print("FATROP fails: ", fatrop_fails)

    if solve_feasibility_problem:
        with open('fatrop_n_eval_f.pkl', 'wb') as f:
            pickle.dump(fatrop_list_n_eval_f, f)
        with open('fatrop_n_eval_g.pkl', 'wb') as f:
            pickle.dump(fatrop_list_n_eval_g, f)
        with open('fatrop_n_eval_gradient_f.pkl', 'wb') as f:
            pickle.dump(fatrop_list_n_eval_gradient_f, f)
        with open('fatrop_n_eval_jacobian_g.pkl', 'wb') as f:
            pickle.dump(fatrop_list_n_eval_jacobian_g, f)
        with open('fatrop_n_eval_hessian_l.pkl', 'wb') as f:
            pickle.dump(fatrop_list_n_eval_hessian_l, f)
        with open('fatrop_t_wall_total.pkl', 'wb') as f:
            pickle.dump(fatrop_list_t_wall_total, f)

###############################################################################
# Run the simulation
###############################################################################
if __name__ == "__main__":
    test_on_cart_pendulum(True)

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
from fp_ddp_python.ocp_transformer import OCP_To_Data_Transformer

###############################################################################
# Test the problems
###############################################################################
def create_nlp_from_ocp(solve_feasibility_problem=False):
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

    nlp = {'x':x, 'p':p, 'f':obj, 'g':g}

    return nlp, x0_vector, lbg, ubg

def test_on_cart_pendulum(solve_feasibility_problem=False, solver_name='ipopt', use_jit=False):

    list_n_eval_f = []
    list_n_eval_g = []
    list_n_eval_gradient_f = []
    list_n_eval_jacobian_g = []
    list_n_eval_hessian_l = []
    list_t_wall_total = []
    list_n_iter = []

    N = 100
    nx = 5
    nu = 1
    n_problems = 100
    n_tests = 20
    nlp, x0_vector, lbg, ubg = create_nlp_from_ocp(solve_feasibility_problem)

    if solver_name == 'fatrop':
        if use_jit:
            opts = {'expand':True, 'structure_detection':'auto',
                                    'debug':False, 'equality': N*nx*[True],
                                    # 'cache':{'nlp_hess_l':feasibility_problem_data['feas_nlp_casadi_gn_hessian']},
                                    'jit':True, 'jit_temp_suffix':False,
                                    'jit_options':{'flags': ['-O3'],'compiler': 'gcc'}}
        else:
            opts = {'expand':True, 'structure_detection':'auto', 'debug':False, 'equality': N*nx*[True]}
        solver = cs.nlpsol('solver', 'fatrop', nlp, opts)
    else:
        if use_jit:
            opts = {'expand':True, 'cache':{'nlp_hess_l':None},
                    'jit':True, 'jit_temp_suffix':False,
                    'jit_options':{'flags': ['-O3'],
                    'compiler': 'gcc'}}
        else:
            opts = {'expand':True}
        solver = cs.nlpsol('solver', 'ipopt', nlp, opts)

    moon_x = cs.linspace(0.7, 4.3, 100)
    fails = []

    # Run the simulation
    n_call_f = 0
    n_call_g = 0
    n_call_grad_f = 0
    n_call_jac_g = 0
    n_call_hess_l = 0
    t_wall_tmp = 0.0
    for i in range(n_problems):

        t_tmp = []
        for _ in range(n_tests):
            res = solver(x0=x0_vector, p=np.array([moon_x[i]]), lbg=lbg, ubg=ubg)

            stats = solver.stats()
            if res['f'] > 1e-8:
                t_tmp.append(np.inf)
                n_eval_f = np.inf
                n_eval_g = np.inf
                n_eval_gradient_f = np.inf
                n_eval_jac_g = np.inf
                n_eval_hess_l = np.inf
                n_iter = np.inf
                fails.append(i)
            else:
                print(solver)
                if solver_name == 'fatrop':
                    t_tmp.append(stats['fatrop']['time_total'])
                    n_eval_f = stats['fatrop']['eval_obj_count']
                    n_eval_g = stats['fatrop']['eval_cv_count']
                    n_eval_gradient_f = stats['fatrop']['eval_grad_count']
                    n_eval_jac_g = stats['fatrop']['eval_jac_count']
                    n_eval_hess_l = stats['fatrop']['eval_hess_count']
                    n_iter = stats['fatrop']['iterations_count']
                elif solver_name == 'ipopt':
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
                    n_iter = stats['iter_count']

        list_t_wall_total.append(min(t_tmp))
        list_n_eval_f.append(n_eval_f)
        list_n_eval_g.append(n_eval_g)
        list_n_eval_gradient_f.append(n_eval_gradient_f)
        list_n_eval_jacobian_g.append(n_eval_jac_g)
        list_n_eval_hessian_l.append(n_eval_hess_l)
        list_n_iter.append(n_iter)

    print("t_wall total: ", list_t_wall_total)
    print("fails: ", fails)

    if solve_feasibility_problem and solver_name == 'fatrop':
        with open('fatrop_n_eval_f.pkl', 'wb') as f:
            pickle.dump(list_n_eval_f, f)
        with open('fatrop_n_eval_g.pkl', 'wb') as f:
            pickle.dump(list_n_eval_g, f)
        with open('fatrop_n_eval_gradient_f.pkl', 'wb') as f:
            pickle.dump(list_n_eval_gradient_f, f)
        with open('fatrop_n_eval_jacobian_g.pkl', 'wb') as f:
            pickle.dump(list_n_eval_jacobian_g, f)
        with open('fatrop_n_eval_hessian_l.pkl', 'wb') as f:
            pickle.dump(list_n_eval_hessian_l, f)
        with open('fatrop_t_wall_total.pkl', 'wb') as f:
            pickle.dump(list_t_wall_total, f)

    if solve_feasibility_problem and not solver_name == 'ipopt':
        with open('ipopt_t_wall_total.pkl', 'wb') as f:
            pickle.dump(list_n_eval_hessian_l, f)
        with open('ipopt_n_eval_f.pkl', 'wb') as f:
            pickle.dump(list_n_eval_f, f)
        with open('ipopt_n_eval_g.pkl', 'wb') as f:
            pickle.dump(list_n_eval_g, f)
        with open('ipopt_n_eval_gradient_f.pkl', 'wb') as f:
            pickle.dump(list_n_eval_gradient_f, f)
        with open('ipopt_n_eval_jacobian_g.pkl', 'wb') as f:
            pickle.dump(list_n_eval_jacobian_g, f)
        with open('ipopt_n_eval_hessian_l.pkl', 'wb') as f:
            pickle.dump(list_n_eval_hessian_l, f)

###############################################################################
# Run the simulation
###############################################################################
if __name__ == "__main__":
    test_on_cart_pendulum(solve_feasibility_problem=True, solver_name='fatrop', use_jit=True)
    test_on_cart_pendulum(solve_feasibility_problem=True, solver_name='ipopt', use_jit=True)

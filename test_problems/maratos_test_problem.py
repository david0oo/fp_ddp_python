from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
import numpy as np
from casadi import *
from matplotlib import pyplot as plt
from itertools import product
# Simplest NLP with Marathos effect
#
# min x_1
#
# s.t. x_1^2 + x_2^2 = 1

# Settings
PLOT = False
FOR_LOOPING = False # call solver in for loop to get all iterates
TOL = 1e-6

def main():
    # run test cases
    params = {'globalization': ['MERIT_BACKTRACKING', 'FUNNEL_METHOD'],
              'line_search_use_sufficient_descent' : [1]}

    keys, values = zip(*params.items())
    for combination in product(*values):
        setting = dict(zip(keys, combination))
        solve_marathos_problem_with_setting(setting)


def solve_marathos_problem_with_setting(setting):

    globalization = setting['globalization']
    line_search_use_sufficient_descent = setting['line_search_use_sufficient_descent']

    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    # set model
    model = AcadosModel()
    x1 = SX.sym('x1')
    x2 = SX.sym('x2')
    x = vertcat(x1, x2)

    # dynamics: identity
    model.disc_dyn_expr = x
    model.x = x
    model.u = SX.sym('u', 0, 0) # [] / None doesnt work
    model.p = []
    model.name = f'marathos_problem'
    ocp.model = model

    # discretization
    Tf = 1
    N = 1
    ocp.dims.N = N
    ocp.solver_options.tf = Tf

    # cost
    ocp.cost.cost_type_e = 'EXTERNAL'
    ocp.model.cost_expr_ext_cost_e = x1

    # constarints
    ocp.model.con_h_expr_0 = x1 ** 2 + x2 ** 2
    ocp.constraints.lh_0 = np.array([1.0])
    ocp.constraints.uh_0 = np.array([1.0])

    # set options
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES
    ocp.solver_options.hessian_approx = 'EXACT'
    ocp.solver_options.integrator_type = 'DISCRETE'
    ocp.solver_options.print_level = 1
    ocp.solver_options.tol = TOL
    ocp.solver_options.nlp_solver_type = 'SQP' # SQP_RTI, SQP
    ocp.solver_options.globalization = globalization
    ocp.solver_options.alpha_min = 1e-2
    ocp.solver_options.levenberg_marquardt = 1e-1
    SQP_max_iter = 100
    ocp.solver_options.qp_solver_iter_max = 400
    ocp.solver_options.regularize_method = 'MIRROR'
    ocp.solver_options.line_search_use_sufficient_descent = line_search_use_sufficient_descent
    ocp.solver_options.eps_sufficient_descent = 1e-1
    ocp.solver_options.qp_tol = 5e-7

    ocp.solver_options.nlp_solver_max_iter = SQP_max_iter
    ocp_solver = AcadosOcpSolver(ocp, json_file=f'{model.name}.json')

    # initialize solver
    rad_init = 0.1 #0.1 #np.pi / 4
    xinit = np.array([np.cos(rad_init), np.sin(rad_init)])
    # xinit = np.array([0.82120912, 0.58406911])
    [ocp_solver.set(i, "x", xinit) for i in range(N+1)]

    # solve
    ocp_solver.solve()
    ocp_solver.print_statistics()
    iter = ocp_solver.get_stats('sqp_iter')
    alphas = ocp_solver.get_stats('alpha')[1:]
    qp_iters = ocp_solver.get_stats('qp_iter')
    residuals = ocp_solver.get_stats('statistics')[1:5,1:iter]

    # get solution
    solution = ocp_solver.get(0, "x")

    # print summary
    print(f"solved Marathos test problem with settings {setting}")
    print(f"cost function value = {ocp_solver.get_cost()} after {iter} SQP iterations")
    print(f"alphas: {alphas[:iter]}")
    print(f"total number of QP iterations: {sum(qp_iters[:iter])}")
    # max_infeasibility = np.max(residuals[1:3])
    # print(f"max infeasibility: {max_infeasibility}")

    # compare to analytical solution
    exact_solution = np.array([-1, 0])
    sol_err = max(np.abs(solution - exact_solution ))

    # checks
    if sol_err > TOL*1e1:
        # raise Exception(f"error of numerical solution wrt exact solution = {sol_err} > tol = {TOL*1e1}")
        print(f"numerical solutions do not match to analytical solution with tolerance {TOL}")
    else:
        print(f"matched analytical solution with tolerance {TOL}")

    if PLOT:
        plt.figure()
        axs = plt.plot(solution[0], solution[1], 'x', label='solution')

        if FOR_LOOPING: # call solver in for loop to get all iterates
            cm = plt.cm.get_cmap('RdYlBu')
            axs = plt.scatter(iterates[:iter+1,0], iterates[:iter+1,1], c=range(iter+1), s=35, cmap=cm, label='iterates')
            plt.colorbar(axs)

        ts = np.linspace(0,2*np.pi,100)
        plt.plot(1 * np.cos(ts)+0,1 * np.sin(ts)-0, 'r')
        plt.axis('square')
        plt.legend()
        plt.title(f"Marathos problem with N = {N}, x formulation")
        plt.show()

    print(f"\n\n----------------------\n")

if __name__ == '__main__':
    main()

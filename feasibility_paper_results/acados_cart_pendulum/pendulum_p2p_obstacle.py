from acados_template import AcadosOcp, AcadosOcpSolver
from time_optimal_pendulum_model import export_free_time_pendulum_ode_model
import numpy as np
import pickle

def create_ocp():
    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    moon_x = 0.7
    # set model
    model = export_free_time_pendulum_ode_model()
    ocp.model = model
    ocp.parameter_values = np.array([moon_x])

    # Parameters
    max_f = 20.
    max_v = 10.
    max_theta = np.pi/4
    
    p_end = 5.0
    Tf = 1.0
    nx = model.x.rows()
    nu = model.u.rows()
    N = 100

    # obstacle position
    moon_midpoint = np.array([moon_x, 0.9])
    moon_radius = 0.3

    pendulum_position = np.array([model.x[1]-0.8*np.sin(model.x[2]), 0.8*np.cos(model.x[2])])

    # set dimensions
    ocp.dims.N = N

    ###########################################################################
    # Define constraints
    ###########################################################################

    # Initial conditions
    ocp.constraints.lbx_0 = np.array([1e-3, 0.0, 0.0, 0.0, 0.0])
    ocp.constraints.ubx_0 = np.array([10.0, 0.0, 0.0, 0.0, 0.0])
    ocp.constraints.idxbx_0 = np.array([0, 1, 2, 3, 4])

    # Actuator constraints
    ocp.constraints.lbu = np.array([-max_f])
    ocp.constraints.ubu = np.array([+max_f])
    ocp.constraints.idxbu = np.array([0])

    # Circular obstacle constraint
    ocp.constraints.uh = np.array([100.0]) # doenst matter
    ocp.constraints.lh = np.array([moon_radius**2])
    x_square = (pendulum_position[0]-model.p[0]) **2 + (pendulum_position[1]-moon_midpoint[1])**2
    ocp.model.con_h_expr = x_square

    # Path constraints
    ocp.constraints.lbx = np.array([-1.0, -max_theta, -max_v])
    ocp.constraints.ubx = np.array([7.0, max_theta, max_v])
    ocp.constraints.idxbx = np.array([1, 2, 3])

    # Terminal constraints
    ocp.constraints.lbx_e = np.array([p_end, 0.0, 0.0, 0.0])
    ocp.constraints.ubx_e = np.array([p_end, 0.0, 0.0, 0.0])
    ocp.constraints.idxbx_e = np.array([1, 2, 3, 4])

    # ------------------- set options -----------------------------------------
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON' # 'GAUSS_NEWTON', 'EXACT'
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.print_level = 0
    ocp.solver_options.nlp_solver_type = 'DDP' # SQP_RTI, SQP
    ocp.solver_options.nlp_solver_max_iter = 100
    ocp.solver_options.globalization = 'MERIT_BACKTRACKING'
    ocp.solver_options.with_adaptive_levenberg_marquardt = True

    # set prediction horizon
    ocp.solver_options.tf = Tf

    ocp.translate_to_feasibility_problem()

    ocp_solver = AcadosOcpSolver(ocp, json_file = 'acados_ocp.json')

    # Remove cost scaling with time steps
    for i in range(N):
        ocp_solver.cost_set(i, "scaling", 1.0)
    simX = np.zeros((N+1, nx))
    simU = np.zeros((N, nu))

    return ocp_solver

if __name__ == '__main__':
    stats = []
    timings = []
    n_iter = []
    n_problems = 100
    N = 100
    n_runs_per_instance = 20
    ocp_solver = create_ocp()
    moon_x = np.linspace(0.7, 4.3, n_problems)
    x_init = np.array([5.0, 0.0, 0.0, 0.0, 0.0])
    u_init = np.array([0.0])


    for i in range(n_problems):
        times = []
        status_acados = []
        for k in range(n_runs_per_instance):
            # Initial guess
            for j in range(N):
                ocp_solver.set(j, "x", x_init)
                ocp_solver.set(j, "u", u_init)
                # Set parameter
                ocp_solver.set(j, "p", np.array([moon_x[i]]))
            ocp_solver.set(N, "x", x_init)
            ocp_solver.set(N, "p", np.array([moon_x[i]]))

            status = ocp_solver.solve()
            times.append(ocp_solver.get_stats("time_tot"))
            status_acados.append(status)
        
        n_iter.append(ocp_solver.get_stats("sqp_iter"))
        timings.append(min(times))
        if max(status_acados) == 0:
            # print("Success")
            stats.append("Success")
        else:
            print("Fail")
            stats.append("Fail")

    with open('acados_n_eval_gn_hessian.pkl', 'wb') as f:
        pickle.dump(n_iter, f)
    with open('acados_timings.pkl', 'wb') as f:
        pickle.dump(timings, f)

    print(stats)
    print(n_iter)
    print(timings)
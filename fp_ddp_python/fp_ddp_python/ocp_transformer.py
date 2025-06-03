# Make available sin, cos, etc
from numpy import *
# Import the project
import rockit
import casadi as cs
import numpy as np


class OCP_To_Data_Transformer:

    def __init__(self) -> None:
        pass

    def detect_boundedness_of_constraints(self,
                                          n_constraints,
                                          lbg,
                                          ubg,
                                          initial_indeces,
                                          dynamic_indeces):

        lower_bounded = []
        lower_upper_bounded = []
        upper_bounded = []
        for i in range(n_constraints):
            # Don't take indeces of initial condition and dynamics
            if i in initial_indeces:
                continue

            if i in dynamic_indeces:
                continue

            # Take initial conditions
            if lbg[i] != -cs.inf:
                if ubg[i] != cs.inf:
                    lower_upper_bounded.append(i)
                else:
                    lower_bounded.append(i)
            elif ubg[i] != cs.inf:
                upper_bounded.append(i)

        return lower_bounded, lower_upper_bounded, upper_bounded

    def retrieve_info_original_nlp(self, data:dict, ocp: rockit.Ocp):

        # Number of constraints and stages
        opts = {"regularity_check":True}
        # opts = {}
        n_variables = ocp._method.opti.x.shape[0]
        n_constraints = ocp._method.opti.g.shape[0]
        N = ocp._method.N

        dynamics_indeces_per_stage = N*[[]]
        control_constraint_indeces_per_stage = N*[[]]
        state_constraints_indeces_per_stage = (N+1)*[[]]
        general_constraints_indeces_per_stage = (N+1)*[[]]
        constraints_per_stage = (N+1) * [[]]

        dynamics_indeces = []
        constraint_indeces = []
        initial_condition_indeces = []
        control_indeces = []

        # New stuff
        states_sampled = ocp.sample(ocp.x, grid="control")[1]
        # states_positions = cs.jacobian(ocp._method.opti.x,states_sampled).sparsity()

        controls_sampled = ocp.sample(ocp.u, grid="control-")[1]
        # control_positions = cs.jacobian(ocp._method.opti.x,control_sampled).sparsity()
        jac_sparsity = cs.jacobian(ocp._method.opti.g,ocp._method.opti.x)

        all_control_positions = []
        for control in ocp.controls:
            control_sampled = ocp.sample(control, grid='control-')[1]
            control_positions = cs.jacobian(ocp._method.opti.x,control_sampled).sparsity().row()
            all_control_positions += control_positions

        all_state_positions = []
        for state in ocp.states:
            state_sampled = ocp.sample(state, grid='control')[1]
            state_positions = cs.jacobian(ocp._method.opti.x,state_sampled).sparsity().row()
            all_state_positions += state_positions

        #######################################################################
        # Retrieve the constraint indeces
        #######################################################################
        for i in range(N+1):

            # there are duplicates since variables are in different constraints several times
            if i < N:
                stage_indeces = cs.jacobian(ocp._method.opti.g,cs.vertcat(states_sampled[:,i], controls_sampled[:,i])).sparsity().row()
            else:
                stage_indeces = cs.jacobian(ocp._method.opti.g, states_sampled[:,i]).sparsity().row()
            # remove the duplicates
            stage_indeces = list(dict.fromkeys(stage_indeces))
            # sort the list
            stage_indeces.sort()
            # remove indeces that belong to previous stage due to dynamics
            if i == N:
                print("Here!")

            if i > 0:
                # Determine the dynamics indeces within the previous stage
                dynamics_indeces_stage = [j for j in constraints_per_stage[i-1] if j in stage_indeces and "F" in str(ocp._method.opti.debug.g_lookup(j))]
                dynamics_indeces_per_stage[i-1] = dynamics_indeces_stage
                # Remove the dynamics indeces from new stage
                stage_indeces = [j for j in stage_indeces if j not in constraints_per_stage[i-1]]
                # Determine dynamics indeces

            # Add constraints to list
            constraints_per_stage[i] = stage_indeces

        #######################################################################
        # Retrieve the stagewise indeces for control, state, and general constraints
        #######################################################################
        for i in range(N+1):
            if i < N:
                stage_indeces = list(set(constraints_per_stage[i]).difference(dynamics_indeces_per_stage[i]))
            else:
                stage_indeces = constraints_per_stage[i]
            control_ind = []
            state_ind = []
            constraint_ind = []

            for index in stage_indeces:
                if jac_sparsity[index,:].nnz() == 1 and jac_sparsity[index,:].sparsity().T.row()[0] in all_control_positions:
                    control_ind.append(index)
                elif jac_sparsity[index,:].nnz() == 1 and jac_sparsity[index,:].sparsity().T.row()[0] in all_state_positions:
                    state_ind.append(index)
                else:
                    constraint_ind.append(index)

            if i < N:
                control_constraint_indeces_per_stage[i] = control_ind

            state_constraints_indeces_per_stage[i] = state_ind
            general_constraints_indeces_per_stage[i] = constraint_ind

        #######################################################################
        # Initial condition
        #######################################################################
        initial_condition_indeces_tmp = state_constraints_indeces_per_stage[0]
        initial_condition_indeces = []
        for index in initial_condition_indeces_tmp:
            if "==" in str(ocp._method.opti.debug.g_lookup(index)):
                initial_condition_indeces.append(index)

        # HPIPM Initial condition indeces
        # Number of rows for matrix
        n_rows = len(cs.jacobian(ocp._method.opti.g[initial_condition_indeces], states_sampled[:,0]).sparsity().row())
        hpipm_initial_condition_positions = cs.jacobian(ocp._method.opti.g[initial_condition_indeces], states_sampled[:,0]).T.sparsity().row()

        combined_positions = list(map(lambda x, y:(x,y), list(range(n_rows)), hpipm_initial_condition_positions))
        hpipm_initial_condition_J_matrix = np.zeros((n_rows, ocp.nx))
        for tup in combined_positions:
            hpipm_initial_condition_J_matrix[tup] = 1.0

        hpipm_initial_condition_bounds = cs.evalf(ocp._method.opti.lbg)[initial_condition_indeces]

        # Indeces for the whole NLP
        for dyn_ind in dynamics_indeces_per_stage:
            dynamics_indeces += dyn_ind

        # non_dynamics_indeces = [index for index in list(range(n_constraints)) if index not in initial_condition_indeces + dynamics_indeces]
        non_dynamics_indeces = [index for index in list(range(n_constraints)) if index not in dynamics_indeces]

        #######################################################################
        # Extract information about NLP from OCP
        #######################################################################
        x_vector = ocp._method.opti.x
        parameters = ocp._method.opti.p
        x0_vector = cs.DM(ocp.initial_value(ocp._method.opti.x))
        p0_vector = cs.DM(ocp.initial_value(ocp._method.opti.p))

        # Original constraint function
        lbg_original = cs.evalf(ocp._method.opti.debug.lbg)
        ubg_original = cs.evalf(ocp._method.opti.debug.ubg)

        # Initialize the initial state with the initial condition depends on the
        # x0_vector[hpipm_initial_condition_positions] = lbg_original[initial_condition_indeces]
        # x0_vector.to_file("x0.mtx")

        original_g_fun = cs.Function("g_fun", [x_vector, parameters], [ocp._method.opti.debug.g], opts)
        original_jac_g_fun = cs.Function("jac_g_fun", [x_vector, parameters], [cs.jacobian(ocp._method.opti.debug.g, x_vector)], opts)

        # Get lower, upper, and lower-upper bounded indeces
        lb_indeces, lb_ub_indeces, ub_indeces =\
            self.detect_boundedness_of_constraints(lbg_original.shape[0],
                                                   lbg_original,
                                                   ubg_original,
                                                   initial_condition_indeces,
                                                   dynamics_indeces)

        #######################################################################
        # Build the dict
        #######################################################################
        data['x0'] = x0_vector
        data['p0'] = p0_vector
        data['n_variables'] = n_variables
        data['n_parameters'] = parameters.shape[0]
        data['n_states_per_stage'] = ocp.nx
        data['n_controls_per_stage'] = ocp.nu
        data['N'] = N
        data['nlp_n_constraints'] = n_constraints
        data['nlp_lbg'] = lbg_original
        data['nlp_ubg'] = ubg_original
        data['nlp_g'] = ocp._method.opti.debug.g
        data['nlp_f'] = ocp._method.opti.debug.f
        data['nlp_x'] = ocp._method.opti.debug.x
        data['nlp_g_fun'] = original_g_fun
        data['nlp_jac_g_fun'] = original_jac_g_fun
        data['nlp_g_initial_condition_indeces'] = initial_condition_indeces
        # data['nlp_g_dynamics_indeces'] = initial_condition_indeces + dynamics_indeces
        data['nlp_g_dynamics_indeces'] = dynamics_indeces
        data['nlp_g_non_dynamics_indeces'] = non_dynamics_indeces
        data['nlp_g_lower_bounded_indeces'] = lb_indeces
        data['nlp_g_lower_upper_bounded_indeces'] = lb_ub_indeces
        data['nlp_g_upper_bounded_indeces'] = ub_indeces

        data['nlp_x_control_positions'] = all_control_positions
        data['hpipm_initial_condition_positions'] = hpipm_initial_condition_positions
        data['hpipm_initial_condition_J_matrix'] = hpipm_initial_condition_J_matrix
        data['hpipm_initial_condition_bounds'] = hpipm_initial_condition_bounds

        return data

    def retrieve_feasibility_problem_info(self, data: dict, ocp: rockit.Ocp, smoothmax=False):

        opts = {"regularity_check":True}
        initial_condition_indeces = data['nlp_g_initial_condition_indeces']
        dynamics_indeces = data['nlp_g_dynamics_indeces']
        constraint_indeces = data['nlp_g_non_dynamics_indeces']
        # Create functions for dynamics
        N = ocp._method.N # More elegant way possible?
        dt = ocp.value(ocp.tf).to_DM()/N
        discrete_dynamics = ocp.discrete_system()
        x_sym = cs.MX.sym('x1', discrete_dynamics.size1_in(0))
        u_sym = cs.MX.sym('u', discrete_dynamics.size1_in(1))
        p0_vector = ocp.initial_value(ocp._method.opti.p)
        discrete_dynamics_fun = cs.Function('discrete_dynamics', [x_sym, u_sym], [discrete_dynamics(x_sym, u_sym, dt, 0, p0_vector, cs.DM.zeros(0))[0]], opts)

        ## Get the indeces of all constraints apart from dynamics!
        # g_constraints = ocp._method.opti.debug.g[initial_condition_indeces + constraint_indeces]
        # lbg_constraints = cs.evalf(ocp._method.opti.debug.lbg[initial_condition_indeces + constraint_indeces])
        # ubg_constraints = cs.evalf(ocp._method.opti.debug.ubg[initial_condition_indeces + constraint_indeces])

        g_constraints = ocp._method.opti.debug.g[constraint_indeces]
        lbg_constraints = cs.evalf(ocp._method.opti.debug.lbg[constraint_indeces])
        ubg_constraints = cs.evalf(ocp._method.opti.debug.ubg[constraint_indeces])

        ## Get dynamics excluding initial condition
        g_dynamics = ocp._method.opti.debug.g[dynamics_indeces]
        lbg_dynamics = cs.evalf(ocp._method.opti.debug.lbg[dynamics_indeces])
        ubg_dynamics = cs.evalf(ocp._method.opti.debug.ubg[dynamics_indeces])

        x_vector = ocp._method.opti.x
        parameters = ocp._method.opti.p
        ## ------------ Define the new objective function -------------------------
        x = cs.MX.sym('x')
        # smoothmax = cs.Function('smoothmax', [x], [cs.logsumexp(cs.vertcat(x,0), cs.MX(0.000001))])
        # smoothmax = cs.Function('smoothmax', [x], [cs.logsumexp(cs.vertcat(x,0), cs.MX(1e-10))])
        # smoothmap = smoothmax.map(g_constraints.shape[0])
        # if smoothmax:
        # # A smooth max version
        #     new_f = 0.5*cs.sumsqr(smoothmap(lbg_constraints-g_constraints))
        #     new_f += 0.5*cs.sumsqr(smoothmap(g_constraints-ubg_constraints))
        # else:
        new_f = 0.5*cs.sumsqr(cs.fmax(0, lbg_constraints-g_constraints))
        new_f += 0.5*cs.sumsqr(cs.fmax(0, g_constraints-ubg_constraints))

        ## ------------ Define functions ------------------------------------------
        # # Function for the dynamics constraints
        # dyn_con_fun = cs.Function("dyn_fun", [x_vector, parameters], [g_dynamics])
        # jac_dyn_con_fun = cs.Function("jac_dyn_fun", [x_vector, parameters], [cs.jacobian(g_dynamics, x_vector)])
        # Function for only dynamics no initial condition
        dyn_con_fun = cs.Function("dyn_fun", [x_vector, parameters], [g_dynamics], opts)
        jac_dyn_con_fun = cs.Function("jac_dyn_fun", [x_vector, parameters], [cs.jacobian(g_dynamics, x_vector)], opts)

        # Functions for objective
        # if smoothmax:
        #     gn_con1 = cs.jacobian(smoothmap(lbg_constraints-g_constraints), x_vector)
        #     gn_con2 = cs.jacobian(smoothmap(g_constraints-ubg_constraints), x_vector)
        # else:
        gn_con1 = cs.jacobian(cs.fmax(0, lbg_constraints-g_constraints), x_vector)
        gn_con2 = cs.jacobian(cs.fmax(0, g_constraints-ubg_constraints), x_vector)
        hessian_fun = cs.Function("hessian_fun", [x_vector, parameters], [gn_con1.T @ gn_con1 + gn_con2.T @ gn_con2], opts)
        # Define a function for Casadi Hessian
        con_multiplier = cs.MX.sym('lam_g', g_dynamics.shape[0])
        obj_multiplier = cs.MX.sym('obj_mult', 1)
        # casadi_hessian_fun = cs.Function("nlp_hess_l", [x_vector, parameters, obj_multiplier, con_multiplier], [cs.gradient(new_f + con_multiplier.T @ g_dynamics,x_vector), gn_con1.T @ gn_con1 + gn_con2.T @ gn_con2, ], opts)
        casadi_hessian_fun = cs.Function("casadi_hess_fun", [x_vector, parameters, obj_multiplier, con_multiplier], [cs.triu(gn_con1.T @ gn_con1 + gn_con2.T @ gn_con2)], opts)
        gradient_obj_fun = cs.Function("gradient_obj_fun", [x_vector, parameters], [cs.gradient(new_f,x_vector)], opts)
        obj_fun = cs.Function("gradient_obj_fun", [x_vector, parameters], [new_f], opts)

        ## ------------- Define objective gradient wrt to controls-------------
        u_vector = cs.MX.sym('controls', N*ocp.nu)
        x_placeholder = cs.MX.sym('x_placeholder', ocp.nx)
        x_tmp = x_placeholder
        new_x_wrt_u_vector = cs.vertcat(x_tmp)
        new_u_from_x = cs.vertcat([])

        for i in range(N):
            new_u_from_x = cs.vertcat(new_u_from_x, x_vector[i*(ocp.nx+ocp.nu)+ocp.nx:(i+1)*(ocp.nx+ocp.nu)])
            u_tmp = u_vector[i*ocp.nu:(i+1)*ocp.nu]

            new_x_wrt_u_vector = cs.vertcat(new_x_wrt_u_vector, u_tmp)
            x_tmp = discrete_dynamics_fun(x_tmp, u_tmp)
            new_x_wrt_u_vector = cs.vertcat(new_x_wrt_u_vector, x_tmp)

        obj_control_tmp = cs.Function("control_obj_tmp", [x_placeholder, u_vector, parameters], [obj_fun(new_x_wrt_u_vector, parameters)])
        obj_control = cs.Function("control_obj", [x_vector, parameters], [obj_control_tmp(x_vector[0:ocp.nx], new_u_from_x, parameters)])
        gradient_control_tmp = cs.Function("control_gradient", [x_placeholder, u_vector, parameters], [cs.gradient(obj_fun(new_x_wrt_u_vector, parameters),u_vector)], opts)

        norm_control_gradient_fun = cs.Function("con_grad", [x_vector, parameters], [gradient_control_tmp(x_vector[0:ocp.nx], new_u_from_x, parameters)], opts)

        ## ------------------- Define data dictionary -----------------------------
        data['feas_nlp_n_constraints'] = g_dynamics.shape[0]
        data['feas_nlp_f_fun'] = obj_fun
        data['feas_nlp_grad_f_fun'] = gradient_obj_fun
        data['feas_nlp_hessian_fun'] = hessian_fun
        data['feas_nlp_f'] = new_f
        data['feas_nlp_g'] = g_dynamics
        data['feas_nlp_lbg'] = lbg_dynamics
        data['feas_nlp_ubg'] = ubg_dynamics
        data['feas_nlp_casadi_gn_hessian'] = casadi_hessian_fun
        data['feas_nlp_g_fun'] = dyn_con_fun
        data['feas_nlp_jac_g_fun'] = jac_dyn_con_fun
        data['feas_nlp_g_initial_condition_indeces'] = initial_condition_indeces
        data['feas_nlp_g_dynamics_indeces'] = dynamics_indeces
        data['feas_nlp_discrete_dynamics_fun'] = discrete_dynamics_fun
        data['feas_nlp_control_obj_fun'] = obj_control
        data['feas_nlp_control_obj_gradient_fun'] = norm_control_gradient_fun

        return data

    def transform(self, ocp: rockit.Ocp, smoothmax=False):
        data = {}
        data = self.retrieve_info_original_nlp(data, ocp)
        data = self.retrieve_feasibility_problem_info(data, ocp, smoothmax)
        return data

###############################################################################
# Class to retrieve an unconstrained feasibility problem
###############################################################################
class OCP_To_Unconstrained_NLP_Transformer():

    def __init__(self) -> None:
        pass

    def transform(self, ocp: rockit.Ocp, smoothmax=False):

        data = {}

        ## Get the indeces of all constraints apart from dynamics!
        g_constraints = ocp._method.opti.debug.g
        lbg_constraints = cs.evalf(ocp._method.opti.debug.lbg)
        ubg_constraints = cs.evalf(ocp._method.opti.debug.ubg)

        x_vector = ocp._method.opti.x
        parameters = ocp._method.opti.p
        x0_vector = cs.DM(ocp.initial_value(ocp._method.opti.x))
        p0_vector = cs.DM(ocp.initial_value(ocp._method.opti.p))
        ## ------------ Define the new objective function -------------------------
        x = cs.MX.sym('x')
        smoothmax = cs.Function('smoothmax', [x], [cs.logsumexp(cs.vertcat(x,0), cs.MX(1e-10))])
        smoothmap = smoothmax.map(g_constraints.shape[0])
        if smoothmax:
        # A smooth max version
            new_f = 0.5*cs.sumsqr(smoothmap(lbg_constraints-g_constraints))
            new_f += 0.5*cs.sumsqr(smoothmap(g_constraints-ubg_constraints))
        else:
            new_f = 0.5*cs.sumsqr(cs.fmax(0, lbg_constraints-g_constraints))
            new_f += 0.5*cs.sumsqr(cs.fmax(0, g_constraints-ubg_constraints))

        ## ------------ Define functions ------------------------------------------
        # Functions for objective
        if smoothmax:
            gn_con1 = cs.jacobian(smoothmap(lbg_constraints-g_constraints), x_vector)
            gn_con2 = cs.jacobian(smoothmap(g_constraints-ubg_constraints), x_vector)
        else:
            gn_con1 = cs.jacobian(cs.fmax(0, lbg_constraints-g_constraints), x_vector)
            gn_con2 = cs.jacobian(cs.fmax(0, g_constraints-ubg_constraints), x_vector)
        hessian_fun = cs.Function("hessian_fun", [x_vector, parameters], [gn_con1.T @ gn_con1 + gn_con2.T @ gn_con2])
        gradient_obj_fun = cs.Function("gradient_obj_fun", [x_vector, parameters], [cs.gradient(new_f,x_vector)])
        obj_fun = cs.Function("gradient_obj_fun", [x_vector, parameters], [new_f])

        ## ------------------- Define data dictionary -----------------------------
        data['feas_nlp_n_variables'] = x_vector.shape[0]
        data['feas_nlp_f_fun'] = obj_fun
        data['feas_nlp_grad_f_fun'] = gradient_obj_fun
        data['feas_nlp_hessian_fun'] = hessian_fun
        data['initial_point'] = x0_vector
        data['parameter_values'] = p0_vector

        return data

if __name__ == "__main__":
    # task_name = "pendulum_balance"
    # deterministic = True
    # task = create_task(task_name)

    load_path = Path.cwd()
    # NOTE: Don't forget to change N and T in the task.py accordingly when loading other problems
    load_path = (
        load_path / "non_converging_iterates"
    )
    cart_on_pole = PendulumOnCartMPC()
    ocp = cart_on_pole.ocp
    ocp_solver = AcadosOcpSolver(ocp)

    files = list(load_path.iterdir())
    print("Number of non convergences in directory: ", len(files) / 3)
    files_grouped = group_files_by_status(files)
    for group in files_grouped.keys():
        files_grouped[group] = sorted(
            files_grouped[group], key=lambda f: extract_status_and_index(f.name)
        )

    assert len(files) == len(files_grouped[1]) + len(files_grouped[2]) + len(
        files_grouped[4]
    ), "Some files have been lost?"

    instance_status = 2

    indices = [1, 211, 469]
    for i in indices:
        run_problem_instance(
            ocp_solver,
            instance_status=instance_status,
            instance_index_ls=i,
            files_grouped=files_grouped,
        )
    # load_and_plot_all(
    #     files_grouped=files_grouped,
    #     identifier="x0",
    # )
    # load_and_plot_all(
    #     files_grouped=files_grouped,
    #     identifier="param",
    # )

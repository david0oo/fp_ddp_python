"""
This file is a wrapper to the HPIPM QP solver.
"""
# Import standard libraries
from __future__ import annotations # <-still need this.
from typing import TYPE_CHECKING
from hpipm_python import *
from hpipm_python.common import *
import casadi as cs
import numpy as np
# Import self-written libraries
from .hpipm_tools import HPIPMConverter
from .iterate import Iterate
if TYPE_CHECKING:
    from .nlp_problem import NLPProblem

cs.DM.set_precision(15)

class HPIPMQPSolver:

    def __init__(self, init_dict: dict) -> None:
        #Converts the QP data into the HPIPM format
        self.dynamics_function = None
        self.hpipm_converter = HPIPMConverter(init_dict)

        self.states_per_stage = init_dict['n_states_per_stage']
        self.controls_per_stage = init_dict['n_controls_per_stage']
        self.N = init_dict['N']

        # For defining the initial state properly
        self.idbx = init_dict['hpipm_initial_condition_positions']
        self.nbx_0 = len(self.idbx)
        self.Jbx = init_dict['hpipm_initial_condition_J_matrix']

        # Dimension object for HPIPM
        self.dim = self.prepare_HPIPM_dim_object()

        # Parameter object
        self.arg = self.prepare_HPIPM_arg_object()

        # qp object
        self.qp_data = hpipm_ocp_qp(self.dim)

        # qp solution object
        self.qp_solution_object = hpipm_ocp_qp_sol(self.dim)

        # set up solver
        self.solver = hpipm_ocp_qp_solver(self.dim, self.arg)

        self.feedback_K = []
        self.feedback_k = []
        self.current_iterate = None

    def set_current_iterate(self, current_iterate: cs.DM):
        """
        Sets the current iterate
        """
        self.x_curr, self.u_curr = self.hpipm_converter.disassemble_iterate(current_iterate)

    def set_discrete_dynamics(self, problem: NLPProblem):
        """
        Sets the current iterate
        """
        self.dynamics_function = problem.discrete_dynamics

    def prepare_HPIPM_dim_object(self):

        dim = hpipm_ocp_qp_dim(self.N)

        dim.set('nx', self.states_per_stage, 0, self.N) # number of states
        dim.set('nu', self.controls_per_stage, 0, self.N-1) # number of inputs

        return dim
    
    def prepare_HPIPM_arg_object(self):

        ## set up solver arg
        mode = 'speed'

        # create and set default arg based on mode
        arg = hpipm_ocp_qp_solver_arg(self.dim, mode)

        # create and set default self.arg based on mode
        arg.set('mu0', 1e4)
        arg.set('iter_max', 100)
        arg.set('tol_stat', 1e-8)
        arg.set('tol_eq', 1e-8)
        arg.set('tol_ineq', 1e-8)
        arg.set('tol_comp', 1e-8)
        # arg.set('reg_prim', 1e-12)

        return arg
    
    def set_qp_data_range(self, qp_data_obj, name, end_index, data):

        for i in range(end_index+1):
            qp_data_obj.set(name, data[i], i)

    def prepare_HPIPM_qp_data(self, jacobian, gradient, hessian, lba, uba, reg_param):

        # Set the dynamics:
        self.As, self.Bs, self.bs = self.hpipm_converter.transform_dynamics_to_stagewise_dynamics(lba,
                                                                                   jacobian)
        # Define the dynamics
        self.set_qp_data_range(self.qp_data, 'A', self.N-1, self.As)
        self.set_qp_data_range(self.qp_data, 'B', self.N-1, self.Bs)
        self.set_qp_data_range(self.qp_data, 'b', self.N-1, self.bs)

        # Set the objective matrices
        self.hessians_states, self.hessians_controls, self.hessians_mixed = self.hpipm_converter.transform_hessian_to_small_hessians(hessian)
        self.gradients_states, self.gradients_controls = self.hpipm_converter.transform_gradient_to_stagewise_gradients(gradient)
        
        # for i in range(len(hessians_controls)):
        #     hessians_controls[i] = hessians_controls[i] + reg_param*np.eye(hessians_controls[i].shape[0])

        self.set_qp_data_range(self.qp_data, 'Q', self.N, self.hessians_states)
        self.set_qp_data_range(self.qp_data, 'S', self.N-1, self.hessians_mixed)
        self.set_qp_data_range(self.qp_data, 'R', self.N-1, self.hessians_controls)
        self.set_qp_data_range(self.qp_data, 'q', self.N, self.gradients_states)
        self.set_qp_data_range(self.qp_data, 'r', self.N-1, self.gradients_controls)
    
    def evaluate_cost(self, x, u, start_index, end_index, calc_end=True):

        cost = 0
        for i in range(start_index, end_index):
            cost += 0.5 * x[i].T @ self.hessians_states[i] @ x[i]
            cost += 0.5 * u[i].T @ self.hessians_controls[i] @ u[i]
            cost += u[i].T @ self.hessians_mixed[i] @ x[i]

            cost += self.gradients_states[i].T @ x[i]
            cost += self.gradients_controls[i].T @ u[i]

        if calc_end:
            cost += 0.5 * x[-1].T @ self.hessians_states[-1] @ x[-1]
            cost += self.gradients_states[-1].T @ x[-1]

        return cost
    
    def alternative_cost(self):
        cost = 0
        grad = cs.vertcat([])
        for i in range(self.N):
            Quu = self.hessians_controls[i] + self.Bs[i].T @ self.feedback_P[i+1] @ self.Bs[i]
            Qu = self.gradients_controls[i] + self.Bs[i].T @ (self.feedback_P[i+1] @ self.bs[i] + self.feedback_p[i+1])
            cost += 0.5 * self.feedback_k[i].T @ Quu @ self.feedback_k[i]

            cost += Qu.T @ self.feedback_k[i]
            grad = cs.vertcat(grad, Qu)

        return cost
    

    def solve_HPIPM_qp(self, jacobian, gradient, hessian, lba, uba, reg_param):

        # Prepare the data
        self.prepare_HPIPM_qp_data(jacobian, gradient, hessian, lba, uba, reg_param)

        # Solve the QP
        self.solver.solve(self.qp_data, self.qp_solution_object)

        # Retrieve the feedback matrices K and k as lists
        self.feedback_K = self.solver.get_feedback(self.qp_data, 'ric_K', 0, self.N-1)
        self.feedback_k = self.solver.get_feedback(self.qp_data, 'ric_k', 0, self.N-1)

        self.feedback_P = self.solver.get_feedback(self.qp_data, 'ric_P', 0, self.N)
        self.feedback_p = self.solver.get_feedback(self.qp_data, 'ric_p', 0, self.N)

        # Retrieve the primal solution
        x = self.qp_solution_object.get('x', 0, self.N)
        self.dx_curr = x
        u = self.qp_solution_object.get('u', 0, self.N-1)

        # cost = self.evaluate_cost(x, u, 0, self.N, True)
        # alt_cost = self.alternative_cost()

        # Retrieve dual solution
        pi = self.qp_solution_object.get('pi', 0, self.N-1)
        lam_lbx = self.qp_solution_object.get('lam_lbx', 0)
        lam_ubx = self.qp_solution_object.get('lam_ubx', 0)
        lam_a_k = - cs.vertcat(cs.DM(lam_lbx-lam_ubx), cs.vertcat(*pi))

        status = self.solver.get('status')

        # Assemble primal solution
        direction_assembled = self.hpipm_converter.assemble_iterate(x, u)
        
        return status, direction_assembled, lam_a_k

    def new_dms_forward_sweep(self, alpha, dx_sol, K, k, u_bar, x_bar):
        xs = [x_bar[0] + alpha*dx_sol[0]]
        # print("Alpha in DMS:", alpha)
        us = []

        for i in range(self.N):
            u_curr = u_bar[i] + alpha*k[i] + K[i] @ (xs[i] - x_bar[i])
            us.append(u_curr)
            x_plus = self.As[i] @ (xs[i] - x_bar[i]) + self.Bs[i] @ (us[i]-u_bar[i]) + alpha*self.bs[i] + x_bar[i+1]
            xs.append(x_plus)

        return xs, us
    
    def new_ddp_forward_sweep(self, alpha, dx_sol, K, k, u_bar, x_bar):
        xs = [x_bar[0] + alpha * dx_sol[0]]
        us = []

        for i in range(self.N):
            u_curr = u_bar[i] + alpha*k[i] + K[i] @ (xs[i] - x_bar[i])
            us.append(u_curr)
            x_plus = self.dynamics_function(xs[i], us[i])
            xs.append(x_plus)

        return xs, us
    
    def dms_multiplier_forward_sweep(self, P, p, x_dms, x_bar):
        lams = []

        for i in range(self.N+1):
            lam = p[i] + P[i] @ (x_dms[i] - x_bar[i])
            lams.append(lam)

        return lams
    
    def create_feasible_ddp_iterate(self, iterate, jacobian, gradient, hessian, lba, uba, alpha, reg_param):
        xs, us = self.hpipm_converter.disassemble_iterate(iterate)

        step_size = alpha
        dx = self.qp_solution_object.get('x', 0, self.N)
        # Prepare the data
        self.prepare_HPIPM_qp_data(jacobian, gradient, hessian, lba, uba, reg_param)

        # Solve the QP
        self.solver.solve(self.qp_data, self.qp_solution_object)

        # Retrieve the feedback matrices K and k as lists
        Ks = self.solver.get_feedback(self.qp_data, 'ric_K', 0, self.N-1)
        ks = self.solver.get_feedback(self.qp_data, 'ric_k', 0, self.N-1)

        sweep_success = False
        while not sweep_success:
            try:
                xs_new, us_new = self.new_ddp_forward_sweep(alpha,
                                                        dx,
                                                        Ks,
                                                        ks,
                                                        us,
                                                        xs)
        
                new_iterate = self.hpipm_converter.assemble_iterate(xs_new,
                                                                    us_new)
                return new_iterate
            except:
                step_size = 0.5*alpha

            if step_size < 1e-17:
                raise RuntimeError("Forward sweep error!")


# Import standard libraries
from __future__ import annotations  # <-still need this.
from typing import TYPE_CHECKING
import casadi as cs
import numpy as np
# Import self-written libraries
if TYPE_CHECKING:
    from .nlp_problem import NLPProblem
    from .iterate import Iterate
    from .parameters import Parameters


class Direction:

    def __init__(self, problem: NLPProblem, parameters: Parameters):
        
        # Initialize the direction
        self.d_k = cs.DM.zeros(problem.number_variables)
        self.lam_a_k = cs.DM.zeros(problem.number_constraints)
        self.G_k = cs.DM.zeros(problem.number_variables)
        self.lba_k = cs.DM.zeros(problem.number_constraints)
        self.uba_k = cs.DM.zeros(problem.number_constraints)
        self.norm_d_k = 0.0
        
        # Regularization parameter
        self.reg_param = 0.0
        self.mu_k = parameters.regularization_mu0#1.0
        self.mu_bar = parameters.regularization_mu_bar#1.0
        self.mu_min = parameters.regularization_mu_min#1e-16
        self.lam = parameters.regularization_lambda#5

        self.alpha_too_small_factor = 0.0
        self.control_positions = problem.nlp_x_control_positions

    def update_mu(self, step_size):

        if step_size == 1.0:
            mu_tmp = self.mu_k
            self.mu_k = cs.fmax(self.mu_min, self.mu_bar/(self.lam))
            self.mu_bar = mu_tmp
        else:
            self.mu_k = cs.fmin(self.lam * self.mu_k, 1.0)

    def eval_quadratic_model_of_objective(self,
                 iterate: Iterate):
        """
        In case of SQP:
        Evaluates the quadratic model of the objective function, i.e.,
        m_k(x_k; p) = grad_f_k.T @ p + p.T @ H_k @ p
        H_k denotes the Hessian Approximation

        In case of SLP:
        Evaluates the linear model of the objective function, i.e.,
        m_k(x_k; p) = grad_f_k.T @ p. This model is used in the termination
        criterion.

        Args:
            p (Casadi DM vector): the search direction where the linear model
                                  should be evaluated

        Returns:
            double: the value of the linear model at the given direction p.
        """
        self.m_k = iterate.f_k + iterate.gradient_f_k.T @ self.d_k + 0.5 * self.d_k.T @ self.H_k @ self.d_k

    def eval_m_k_with_dk(self,
                 iterate: Iterate, d_k: cs.DM):
        """
        In case of SQP:
        Evaluates the quadratic model of the objective function, i.e.,
        m_k(x_k; p) = grad_f_k.T @ p + p.T @ H_k @ p
        H_k denotes the Hessian Approximation

        In case of SLP:
        Evaluates the linear model of the objective function, i.e.,
        m_k(x_k; p) = grad_f_k.T @ p. This model is used in the termination
        criterion.

        Args:
            p (Casadi DM vector): the search direction where the linear model
                                  should be evaluated

        Returns:
            double: the value of the linear model at the given direction p.
        """
        return iterate.f_k + iterate.gradient_f_k.T @ d_k + 0.5 * d_k.T @ self.H_k @ d_k
    
    def linearized_l1_constration_violation(self, d_k):
        return  cs.sum1(cs.fabs(cs.fmax(0, self.lba_k-self.A_k@d_k))) +\
                cs.sum1(cs.fabs(cs.fmax(0, self.A_k@d_k-self.uba_k)))

    def evaluate_model_merit_function(self, penalty, d_k, f_k):
        lin_l1_infeasibility = self.linearized_l1_constration_violation(d_k)
        # print("Linearized l1- constraint violation: ", lin_l1_infeasibility)
        return 0.5 * d_k.T @ self.H_k @ d_k  + self.G_k.T @ d_k + f_k + penalty*lin_l1_infeasibility

    def prepare_qp_data(self, iterate: Iterate, problem: NLPProblem):
        """
        Prepares the objective vector g and the constraint matrix A for the LP.
        I.e., min_{x}   g_k.T x
              s.t.  lba <= A_k*x <= uba,
                    lbx <= x <= ubx.
        """
        self.A_k = iterate.jacobian_g_k
        self.G_k[:] = iterate.gradient_f_k

        gamma = cs.fmin(2*iterate.f_k, 1e-3)

        self.reg_param = self.mu_k*gamma
        self.H_k = iterate.hessian_lagrangian_k + self.reg_param*cs.DM.eye(self.d_k.shape[0])
        self.prepare_qp_bounds(iterate, problem)

    def increase_alpha_factor(self):
        self.alpha_too_small_factor += 10.0

    def reset_alpha_factor(self):
        self.alpha_too_small_factor = 0.0

    def prepare_qp_bounds(self, iterate: Iterate, problem: NLPProblem):
        """
        Prepare the bounds for the constraints in the LP.
        I.e., min_{x}   g_k.T x
              s.t.  lba <= A_k*x <= uba,
                    lbx <= x <= ubx.
        Linearizing the constraints gives a constant part that needs to be
        added to the bounds.
        """
        self.lba_k[:] = problem.lbg - iterate.g_k
        self.uba_k[:] = problem.ubg - iterate.g_k
        
    
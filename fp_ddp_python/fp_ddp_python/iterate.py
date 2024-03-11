"""
This file defines the parent class iterate
"""
# Import standard libraries
from __future__ import annotations # <-still need this.
from typing import TYPE_CHECKING
import casadi as cs
import numpy as np
# Import self-written libraries
if TYPE_CHECKING:
    from .nlp_problem import NLPProblem
    from .logger import Logger
    from .parameters import Parameters


class Iterate:

    def __init__(self,
                 problem: NLPProblem,
                 parameters: Parameters):
        """
        Constructor.
        """
        self.number_variables = problem.number_variables
        self.number_constraints = problem.number_constraints
        self.number_parameters = problem.number_parameters

        self.x_k = cs.DM.zeros(self.number_variables, 1)
        self.g_k = cs.DM.zeros(self.number_constraints, 1)
        self.f_k = 0
        self.gradient_f_k = cs.DM.zeros(self.number_variables,1)
        self.lam_g_k = cs.DM.zeros(self.number_constraints, 1)
        self.p_k = cs.DM.zeros(self.number_parameters, 1)
        self.penalty = parameters.penalty#1.0

    def initialize(self, initialization_dict: dict):

        # # Define iterative variables
        if 'x0' in initialization_dict:
            self.x_k[:] = initialization_dict['x0']

        if 'lam_g0' in initialization_dict:
            self.lam_g_k[:] = initialization_dict['lam_g0']

        if 'p0' in initialization_dict:
            self.p_k[:] = initialization_dict['p0']

    def evaluate_quantities(self,
                            nlp_problem: NLPProblem,
                            log: Logger,
                            step_accepted:False):
        
        # Evaluate functions
        if not step_accepted:
            self.f_k = nlp_problem.eval_f(self.x_k, self.p_k, log)
        self.g_k[:] = nlp_problem.eval_g(self.x_k, self.p_k, log) 
        self.gradient_f_k[:] = nlp_problem.eval_gradient_f(self.x_k, self.p_k, log)
        self.jacobian_g_k = nlp_problem.eval_jacobian_g(self.x_k, self.p_k, log)
        self.gradient_lagrangian_k = nlp_problem.eval_gradient_lagrangian(self.gradient_f_k, self.jacobian_g_k, self.lam_g_k, log)
        self.hessian_lagrangian_k = nlp_problem.eval_hessian_lagrangian(self.x_k, self.p_k, log)

        self.norm_obj_gradient_controls = cs.norm_inf(nlp_problem.norm_control_gradient_fun(self.x_k, self.p_k))
        self.control_obj = nlp_problem.obj_control_fun(self.x_k, self.p_k)

        nlp_g = nlp_problem.eval_nlp_g(self.x_k, self.p_k, log)
        self.nlp_infeasibility = self.feasibility_measure(nlp_g, nlp_problem.nlp_lbg, nlp_problem.nlp_ubg)
        self.infeasibility = self.feasibility_measure(self.g_k, nlp_problem.lbg, nlp_problem.ubg)
        self.stationarity = self.stationarity_condition()

    def stationarity_condition(self):
        """
        Evaluates the stationarity condition, i.e., the norm of the gradient of
        the Lagrangian.
        """
        return float(cs.norm_inf(self.gradient_lagrangian_k))

    def feasibility_measure(self,
                            g_x: cs.DM,
                            lbg: cs.DM,
                            ubg: cs.DM):
        """
        The feasibility measure in the l-\\infty norm.

        Args:
            x (DM-array): value of state variable
            g_x (DM-array): value of constraints at state variable

        Returns:
            double: the feasibility in the l-\\infty norm
        """
        return float(cs.norm_inf(cs.vertcat(
                     cs.fmax(0, lbg-g_x),
                     cs.fmax(0, g_x-ubg))))
    
    def l1_constration_violation(self, g_k: cs.DM, nlp_problem: NLPProblem):
        return  cs.sum1(cs.fabs(cs.fmax(0, nlp_problem.lbg-g_k))) +\
                cs.sum1(cs.fabs(cs.fmax(0, g_k-nlp_problem.ubg)))

    def evaluate_merit_function(self,penalty, f_k, g_k, nlp_problem: NLPProblem):
        l1_viol = self.l1_constration_violation(g_k, nlp_problem)
        return f_k + penalty*l1_viol
    
    def set_penalty(self, value:float):
        self.penalty = value

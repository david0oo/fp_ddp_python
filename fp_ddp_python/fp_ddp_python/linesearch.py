"""
A line search class 
"""
from __future__ import annotations  # <-still need this.
from typing import TYPE_CHECKING

import numpy as np
import casadi as cs

cs.DM.set_precision(15)
# Import self-written libraries
if TYPE_CHECKING:
    from .hpipm_qp_solver import HPIPMQPSolver
    from .iterate import Iterate
    from .direction import Direction
    from .nlp_problem import NLPProblem
    from .logger import Logger
    from .parameters import Parameters


class LineSearch:

    def __init__(self, parameters: Parameters) -> None:
        
        self.alpha_reduction = parameters.ls_alpha_reduction#0.5
        self.eta = parameters.ls_eta#1e-6
        self.alpha_min = parameters.ls_alpha_min#1e-17
        self.alpha = 0.0

    def single_shooting_backtracking_linesearch(self, hpipm_solver: HPIPMQPSolver, nlp_problem:NLPProblem, log: Logger, iterate: Iterate, direction: Direction):
        """
        A globalization based on Single Shooting.
        """
        # print("Current iterate: ", iterate.x_k)
        x_bar, u_bar = nlp_problem.disassemble_iterate(iterate.x_k)

        Ks = hpipm_solver.feedback_K
        ks = hpipm_solver.feedback_k
        dx = hpipm_solver.dx_curr

        direction.eval_quadratic_model_of_objective(iterate)
        pred = iterate.f_k - direction.m_k
        alpha = 1.0
        while True:
            xs, us = hpipm_solver.new_dms_forward_sweep(alpha, dx, Ks, ks, u_bar, x_bar)
            new_states = nlp_problem.forward_dynamics_simulation(xs[0], us)

            trial_x = nlp_problem.assemble_iterate(new_states, us)
            trial_f = nlp_problem.eval_f(trial_x, iterate.p_k, log)
            negative_ared = trial_f - iterate.f_k 
            if negative_ared <= cs.fmin(-self.eta * alpha * max(pred, 0) + 1e-18, 0):
                iterate.x_k = trial_x
                iterate.f_k = trial_f
                iterate.lam_g_k = direction.lam_a_k
                direction.norm_d_k = float(cs.norm_inf(direction.d_k))
                self.alpha = alpha
                direction.update_mu(alpha)
                return

            if alpha < self.alpha_min:
                raise Exception('Error: alpha too small')
            
            # Reduction of penalty parametermia
            alpha = self.alpha_reduction*alpha


    def ddp_backtracking_linesearch(self, hpipm_solver: HPIPMQPSolver, nlp_problem:NLPProblem, log: Logger, iterate: Iterate, direction: Direction):
        """
        A globalization based on Single Shooting.
        """
        x_bar, u_bar = nlp_problem.disassemble_iterate(iterate.x_k)

        Ks = hpipm_solver.feedback_K
        ks = hpipm_solver.feedback_k
        dx = hpipm_solver.dx_curr

        direction.eval_quadratic_model_of_objective(iterate)
        pred = iterate.f_k - direction.m_k
        alpha = 1.0
        while True:
            forward_sweep_success = False
            try: 
                xs, us = hpipm_solver.new_ddp_forward_sweep(alpha, dx, Ks, ks, u_bar, x_bar)
                forward_sweep_success = True
            except:
                print("Forward sweep failed with alpha =" + str(alpha) + "!!")

            if forward_sweep_success:
                trial_x = nlp_problem.assemble_iterate(xs, us)
                d_k = trial_x - iterate.x_k
                m_k = direction.eval_m_k_with_dk(iterate, d_k)
                pred = iterate.f_k - m_k

                trial_f = nlp_problem.eval_f(trial_x, iterate.p_k, log)
                negative_ared = trial_f - iterate.f_k 
                
                if negative_ared <= cs.fmin(-self.eta * max(pred, 0) + 1e-18, 0):
                    direction.reset_alpha_factor()
                    iterate.x_k = trial_x
                    iterate.f_k = trial_f
                    iterate.lam_g_k = direction.lam_a_k
                    direction.norm_d_k = float(cs.norm_inf(direction.d_k))
                    self.alpha = alpha
                    direction.update_mu(alpha)
                    return
                
            if alpha < self.alpha_min:
                raise Exception('Error: alpha too small')
            
            # Reduction of step size
            alpha = self.alpha_reduction*alpha
            
    def dms_backtracking_linesearch(self, hpipm_solver: HPIPMQPSolver, nlp_problem:NLPProblem, log: Logger, iterate: Iterate, direction: Direction):

        current_merit = iterate.evaluate_merit_function(iterate.penalty, iterate.f_k, iterate.g_k, nlp_problem)
        model_value = direction.evaluate_model_merit_function(iterate.penalty, direction.d_k, iterate.f_k)

        pred = current_merit - model_value # iterate.f_k?

        x_bar, u_bar = nlp_problem.disassemble_iterate(iterate.x_k)

        Ks = hpipm_solver.feedback_K
        ks = hpipm_solver.feedback_k
        dx = hpipm_solver.dx_curr

        # Check for high enough penalty parameter
        while True:
            lhs = (current_merit - model_value)
            rhs = 0.1*iterate.penalty * iterate.l1_constration_violation(iterate.g_k, nlp_problem)
            if lhs >= rhs:
                break
            else:
                iterate.penalty *= 10
                current_merit = iterate.evaluate_merit_function(iterate.penalty, iterate.f_k, iterate.g_k, nlp_problem)
                model_value = direction.evaluate_model_merit_function(iterate.penalty, direction.d_k, iterate.f_k)
                pred = current_merit - model_value

        alpha = 1.0
        while True:
            xs, us = hpipm_solver.new_dms_forward_sweep(alpha, dx, Ks, ks, u_bar, x_bar)

            trial_x = nlp_problem.assemble_iterate(xs, us)
            trial_f = nlp_problem.eval_f(trial_x, iterate.p_k, log)
            trial_g = nlp_problem.eval_g(trial_x, iterate.p_k, log)
            trial_merit = iterate.evaluate_merit_function(iterate.penalty, trial_f, trial_g, nlp_problem)
            negative_ared = trial_merit - current_merit 

            if negative_ared <= cs.fmin(-self.eta * alpha * max(pred, 0) + 1e-18, 0):
                iterate.x_k = trial_x
                iterate.f_k = trial_f
                iterate.lam_g_k = direction.lam_a_k
                direction.norm_d_k = float(cs.norm_inf(direction.d_k))
                self.alpha = alpha
                direction.update_mu(alpha)
                return

            if alpha < self.alpha_min:
                raise Exception('Error: alpha too small')
            
            # Reduction of penalty parametermia
            alpha = self.alpha_reduction*alpha
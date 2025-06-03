"""
This file contains the parameters for the Feasible Sequential Linear/Quadratic
Programming solver
"""
from __future__ import annotations  # <-still need this.
from typing import TYPE_CHECKING

import casadi as cs


class Parameters:

    def __init__(self, opts: dict = {}):
        """Initializes the parameters of the solver.

        Raises:
            KeyError: _description_
            KeyError: _description_
            KeyError: _description_
        """
        if not type(opts) is dict:
            raise TypeError("opts must be a dictionary!!")

        #######################################################################
        # Parameters for Termination Criterion
        #######################################################################
        # Solver tolerances for termination -----------------------------------
        if 'convergence_tol' in opts:
            self.convergence_tol = opts['convergence_tol']
        else:
            self.convergence_tol = 1e-8

        if 'infeasibility_tol' in opts:
            self.infeasibility_tol = opts['infeasibility_tol']
        else:
            self.infeasibility_tol = 1e-6

        if 'objective_tol' in opts:
            self.objective_tol = opts['objective_tol']
        else:
            self.objective_tol = 1e-12

        if 'small_step_tol' in opts:
            self.small_step_tol = opts['small_step_tol']
        else:
            self.small_step_tol = 1e-10

        # Max Iterations ------------------------------------------------------
        if 'max_iter' in opts:
            self.max_iter = opts['max_iter']
        else:
            self.max_iter = 1000

        #######################################################################
        # Parameters for Line Search
        #######################################################################
        if 'ls_eta' in opts:
            self.ls_eta = opts['ls_eta']
        else:
            self.ls_eta = 1e-6

        if 'ls_alpha_min' in opts:
            self.ls_alpha_min = opts['ls_alpha_min']
        else:
            self.ls_alpha_min = 1e-17

        if 'ls_alpha_reduction' in opts:
            self.ls_alpha_reduction = opts['ls_alpha_reduction']
        else:
            self.ls_alpha_reduction = 0.5


        #######################################################################
        # Parameters for Direction: Regularization
        #######################################################################
        # Max Iterations ------------------------------------------------------
        if 'regularization_mu_bar' in opts:
            self.regularization_mu_bar = opts['regularization_mu_bar']
        else:
            self.regularization_mu_bar = 1.0

        if 'regularization_mu_min' in opts:
            self.regularization_mu_min = opts['regularization_mu_min']
        else:
            self.regularization_mu_min = 1e-16

        if 'regularization_lambda' in opts:
            self.regularization_lambda = opts['regularization_lambda']
        else:
            self.regularization_lambda = 5

        if 'regularization_mu0' in opts:
            self.regularization_mu0 = opts['regularization_mu0']
        else:
            self.regularization_mu0 = 1e-3#1.0

        #######################################################################
        # Parameters for Iterate
        #######################################################################
        if 'penalty' in opts:
            self.penalty = opts['penalty']
        else:
            self.penalty = 1.0

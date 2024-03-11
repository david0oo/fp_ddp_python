"""
A logger class thats keeps track of the iterates.    
"""
# Import standard libraries
from __future__ import annotations  # <-still need this.
from typing import TYPE_CHECKING
# Import self-written libraries
if TYPE_CHECKING:
    from .iterate import Iterate
    from .direction import Direction

class Logger:
    """
    Logger class that keeps track of interesting stats in the algorithm
    """
    def __init__(self):
        """
        Constructor
        """
        self.reset()
        
    def reset(self):
        # Solver success
        self.solver_success = False
        # number of function evaluations
        self.n_eval_f = 0
        # number of constraint evaluations
        self.n_eval_g = 0
        # number of nlp constraint evaluations
        self.n_eval_nlp_g = 0
        # number of gradient of objective evaluations
        self.n_eval_gradient_f = 0
        # number of Jacobian of constraints evaluations
        self.n_eval_jacobian_g = 0
        # number of Jacobian of constraints evaluations
        self.n_eval_nlp_jacobian_g = 0
        # number of Gradient of Lagrangian evaluations
        self.n_eval_gradient_lagrangian = 0
        # number of Hessian of Lagrangian evaluations
        self.n_eval_hessian_lagrangian = 0
        # number of outer iterations
        self.iteration_counter = 0
        # convergence status of the FSLP algorithm
        self.solver_status = "Not Converged"
        # List of iterates et cetera
        self.list_primal_variables = []
        self.list_times = []
        self.nlp_infeasibility = []
        self.infeasibility = []
        self.stationarity = []
        self.step_sizes = []
        self.kkt_residual = []

        # Timing
        self.t_wall = -1.0

    def increment_n_eval_f(self):
        self.n_eval_f += 1

    def increment_n_eval_g(self):
        self.n_eval_g += 1

    def increment_n_eval_nlp_g(self):
        self.n_eval_nlp_g += 1

    def increment_n_eval_gradient_f(self):
        self.n_eval_gradient_f += 1

    def increment_n_eval_jacobian_g(self):
        self.n_eval_jacobian_g += 1

    def increment_n_eval_nlp_jacobian_g(self):
        self.n_eval_jacobian_g += 1

    def increment_n_eval_gradient_lagrangian(self):
        self.n_eval_gradient_lagrangian += 1

    def increment_n_eval_hessian_lagrangian(self):
        self.n_eval_hessian_lagrangian += 1

    def increment_iteration_counter(self):
        self.iteration_counter += 1

    def set_solver_status(self, status_string):
        self.solver_status = status_string

    def add_iteration_stats(self, iterate: Iterate, direction: Direction):
        self.nlp_infeasibility.append(iterate.nlp_infeasibility)
        self.infeasibility.append(iterate.infeasibility)
        self.stationarity.append(iterate.stationarity)
        self.list_primal_variables.append(iterate.x_k)
        self.step_sizes.append(direction.norm_d_k)
        self.kkt_residual.append(max(iterate.stationarity, iterate.infeasibility))

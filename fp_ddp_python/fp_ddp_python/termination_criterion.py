"""
This is the termination criterion
"""
from __future__ import annotations  # <-still need this.
from typing import TYPE_CHECKING

import casadi as cs
# Import self-written libraries
if TYPE_CHECKING:
    from .iterate import Iterate
    from .direction import Direction
    from .parameters import Parameters

class TerminationCriterion:

    def __init__(self, parameters: Parameters) -> None:
        self.termnination_status = "Not_Converged"
        self.termination_message = "Not Converged"
        self.success = False

        self.infeasibility_tol =  parameters.infeasibility_tol#1e-6
        self.small_step_tol =  parameters.small_step_tol#1e-10
        self.convergence_tol = parameters.convergence_tol#1e-8#1e-6
        self.objective_tol = parameters.objective_tol#1e-12
        self.max_iter = parameters.max_iter#1000

    def print_termination_message(self):
        print(self.termination_message)

    def check_small_step(self, n_iter: int, norm_d_k: float, infeasibility: float):
        if n_iter > 0 and norm_d_k < self.infeasibility_tol:
            if infeasibility < self.infeasibility_tol:
                self.termination_status = "Converged_To_Feasible_Non_OptimalPoint"
                self.termination_message = "Stopped: Converged To Feasible Point"
            else:
                self.termination_status = "Converged_To_Infeasible_Point"
                self.termination_message = "Stopped: Converged To Infeasible Point"
            self.success = False
            return True
        else:
            return False

    def check_stationarity(self, infeasibility: float, stationarity: float):
        if cs.fmax(infeasibility, stationarity) < self.convergence_tol:
            self.termination_status = "Converged_To_KKT_Point"
            self.termination_message = "Optimal Solution found! Converged To KKT Point"
            self.success = True
            return True
        else: 
            return False
        
    def check_max_iterations(self, n_iter):
        if n_iter == self.max_iter:
            self.termination_status = "Maximum_Iterations_Reached"
            self.termination_message = "Stopped: Maximum Iterations Reached"
            self.success = False
            return True
        else:
            return False
        
    def check_objective_value(self, objective_value, infeasibility):
        if infeasibility < self.infeasibility_tol and objective_value < self.objective_tol:
            self.termination_status = "Converged_To_Zero_Residual_Solution"
            self.termination_message = "Optimal Solution found! Converged To Zero Residual Solution"
            self.success = True
            return True
        else:
            return False

    def check_termination(self, n_iter, iterate:Iterate, direction:Direction):
        """
        Checks for the termination of the solver
        """
        # Check if a KKT point was found
        if self.check_stationarity(iterate.infeasibility, iterate.stationarity)\
            or self.check_small_step(n_iter, direction.norm_d_k, iterate.infeasibility)\
            or self.check_max_iterations(n_iter)\
            or self.check_objective_value(iterate.f_k, iterate.infeasibility):
            return True
        else:
            return False    

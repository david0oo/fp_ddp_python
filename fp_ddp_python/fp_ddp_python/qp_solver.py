""" 
This class determines what subproblem is solved. Either QP or LP
"""
# Import standard libraries
from __future__ import annotations # <-still need this.
from typing import TYPE_CHECKING
import casadi as cs
import numpy as np
# Import self-written libraries
from .hpipm_qp_solver import HPIPMQPSolver
if TYPE_CHECKING:
    from .direction import Direction
    from .nlp_problem import NLPProblem
    from .iterate import Iterate


class QPSolver:

    def __init__(self,
                 nlp_problem: NLPProblem,
                #  parameters: Options,
                 feasibility_problem_data: dict):

        # x0 = feasibility_problem_data['x0']
        # p0 = feasibility_problem_data['p0']
        # jac_g = nlp_problem.jacobian_g_function(x0, p0)
        # hessian = nlp_problem.hessian_lagrangian_function(x0, p0)
        # g = nlp_problem.g_function(x0, p0)

        # self.lbg = nlp_problem.lbg
        # self.ubg = nlp_problem.ubg

    	# Create HPIPM QP solver
        self.hpipm_qp_solver = HPIPMQPSolver(feasibility_problem_data)
        self.hpipm_qp_solver.set_discrete_dynamics(nlp_problem)


    def solve_hpipm_qp(self, direction: Direction, iterate: Iterate):
        """
        Using the HPIPM Wrapper, we solve a QP.
        """
        self.hpipm_qp_solver.set_current_iterate(iterate.x_k)

        status, d_k, pi = self.hpipm_qp_solver.solve_HPIPM_qp(direction.A_k,
                                                          direction.G_k,
                                                          direction.H_k,
                                                          direction.lba_k,
                                                          direction.uba_k,
                                                          direction.reg_param)
        return status, d_k, pi

    def create_feasible_ddp_iterate(self,
                                    direction: Direction,
                                    iterate: Iterate,
                                    alpha: float):
        new_x_k = self.hpipm_qp_solver.create_feasible_ddp_iterate(iterate.x_k,
                                                                   direction.A_k,
                                                                   direction.G_k,
                                                                   direction.H_k,
                                                                   direction.lba_k,
                                                                   direction.uba_k,
                                                                   alpha,
                                                                   direction.reg_param)
        
        return new_x_k

    def solve_qp(self, direction: Direction, iterate: Iterate):
        """
        This function solves the qp subproblem. Additionally some processing of
        the result is done and the statistics are saved. The input signature is
        the same as for a casadi qp solver.
        Input:
        g       Vector in QP objective
        lba     lower bounds of constraints
        uba     upper bounds of constraints
        lbx     lower bounds of variables
        ubx     upper bounds of variables

        Return:
        solve_success   Bool, indicating if subproblem was succesfully solved
        p_scale         Casadi DM vector, the new search direction
        lam_p_scale     Casadi DM vector, the lagrange multipliers for the new
                        search direction
        """
        ## Solve QP with HPIPM ------------------------------------------------
        status_tmp, d_tmp, lam_tmp = self.solve_hpipm_qp(direction, iterate)
        # print("HPIPM status: ", status_tmp)
        if status_tmp == 0:
            status_tmp = True
        else:
            print("HPIPM failed! Status: ", status_tmp)
            status_tmp = False

        return status_tmp, d_tmp, lam_tmp
    
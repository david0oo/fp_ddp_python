"""
This file handles the output of the Feasibility Solver
"""
# Import standard libraries
from __future__ import annotations # <-still need this.
from typing import TYPE_CHECKING
import casadi as cs
import numpy as np
# Import self-written libraries
if TYPE_CHECKING:
    from .solver import FeasibilityProblemSolver
    from .iterate import Iterate
    from .direction import Direction
    from .linesearch import LineSearch


class Output:

    def __init__(self):
        pass

    def print_header(self):
        """
        This is the algorithm header
        """
        print("------------------------------------------------------------------------------------")
        print("                            This is the Feasibility Solver                          ")
        print("                         © MECO Research Team, KU Leuven, 2024                      ")
        print("                         © SYSCOP, University of Freiburg, 2024                     ")
        print("------------------------------------------------------------------------------------")

    def print_iteration_header(self):
        """ 
        Prints the iteration header.
        """
        print(("{iter:>6} | {obj:^10} | {inf:^10} | {stat:^10} | "
                   "{alpha:^10} | {d_norm:^10} | {gamma:^10} |{orig_infeas:^15}").format(
                        obj='objective',
                        iter='iter.', inf='infeas.', stat='statio.',
                        orig_infeas='nlp infeas.', alpha='alpha', d_norm='||d||',
                        gamma='reg.'))

    def print_iteration(self,
                     solver: FeasibilityProblemSolver,
                     iterate: Iterate,
                     direction: Direction,
                     i: int,
                     linesearch: LineSearch):
        """
        This function prints the iteration output to the console.

        Args:
            i (integer): the iteration index of the solve operator.
        """
        if i % 10 == 0:
            self.print_iteration_header()

        print(("{iter:>6} | {obj:^10.4e} | {inf:^10.4e} | {stat:^10.4e} | "
                   "{alpha:^10.4e} | {d_norm:^10.4e} | {gamma:^10.4e} | {orig_infeas:^15.4e}").format(
                     obj=float(iterate.f_k),
                     iter=i,
                     inf=iterate.infeasibility,
                     stat=iterate.stationarity,
                     orig_infeas=iterate.nlp_infeasibility,
                     alpha=linesearch.alpha,
                     gamma=float(direction.reg_param),
                     d_norm=direction.norm_d_k))                

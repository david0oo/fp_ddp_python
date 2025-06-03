"""
A prototypical implementation of a feasible solver in Python using

- HPIPM as QP solver
  - from Casadi
  OR
  - from HPIPM_python
- Rockit as tool to model the OCP
- Casadi for function evaluations
"""
import casadi as cs

from .output import Output
from .nlp_problem import NLPProblem
from .iterate import Iterate
from .direction import Direction
from .logger import Logger
from .qp_solver import QPSolver
from .termination_criterion import TerminationCriterion
from .linesearch import LineSearch
from .parameters import Parameters

class FeasibilityProblemSolver:

    def __init__(self,
                 feasibility_problem_data: dict,
                 mode: str = "ddp",
                 init_feasible: bool = False,
                 opts = {}):

        self.mode = mode
        self.alpha = 0.0
        self.n_iter = 0
        self.init_feasible = init_feasible

        # Setup NLP
        self.output = Output()
        self.log = Logger()
        self.parameters = Parameters(opts)
        self.nlp_problem = NLPProblem(feasibility_problem_data)
        self.terminator = TerminationCriterion(self.parameters)
        self.linesearch = LineSearch(self.parameters)
        #
        self.iterate = Iterate(self.nlp_problem, self.parameters)
        self.direction = Direction(self.nlp_problem, self.parameters)
        self.qp_solver = QPSolver(self.nlp_problem, feasibility_problem_data)

    def create_feasible_initial_guess(self):
        self.iterate.evaluate_quantities(self.nlp_problem, self.log, False)
        self.direction.prepare_qp_data(self.iterate, self.nlp_problem)
        if self.iterate.infeasibility >= self.terminator.infeasibility_tol:
            self.iterate.x_k = self.qp_solver.create_feasible_ddp_iterate(self.direction,
                                                                        self.iterate,
                                                                        1.0)

    def solve(self, init_dict: dict):
        """
        Main Optimization loop.
        """
        #######################################################################
        # PREPROCESSING
        #######################################################################
        self.output.print_header()

        self.iterate.initialize(init_dict)
        if self.init_feasible:
            self.create_feasible_initial_guess()

        #######################################################################
        # MAIN OPTIMIZATION LOOP
        #######################################################################
        self.n_iter = 0
        self.step_accepted = False

        while True:

            # Evaluate the functions at x_k
            self.iterate.evaluate_quantities(self.nlp_problem, self.log, self.step_accepted)
            self.log.add_iteration_stats(self.iterate, self.direction)
            self.output.print_iteration(self, self.iterate, self.direction, self.n_iter, self.linesearch)

            if self.terminator.check_termination(self.n_iter, self.iterate, self.direction):
                self.terminator.print_termination_message()
                return

            self.direction.prepare_qp_data(self.iterate, self.nlp_problem)
            solve_success, self.direction.d_k[:], self.direction.lam_a_k[:] = self.qp_solver.solve_qp(self.direction, self.iterate)

            if not solve_success:
                raise RuntimeError('Error in QP solver, this should not happen!')

            ###################################################################
            # GLOBALIZATION
            ###################################################################

            # self.iterate.penalty = .001
            if self.mode == "sqp":
                # Do backtracking line search here
                self.linesearch.sqp_backtracking_linesearch(self.qp_solver.hpipm_qp_solver,
                                                        self.nlp_problem,
                                                        self.log,
                                                        self.iterate,
                                                        self.direction)
            elif self.mode == "dms":
                self.linesearch.dms_backtracking_linesearch(self.qp_solver.hpipm_qp_solver,
                                                        self.nlp_problem,
                                                        self.log,
                                                        self.iterate,
                                                        self.direction)
            elif self.mode == "dss":
                # Do backtracking line search here
                self.linesearch.single_shooting_backtracking_linesearch(self.qp_solver.hpipm_qp_solver,
                                                        self.nlp_problem,
                                                        self.log,
                                                        self.iterate,
                                                        self.direction)
            elif self.mode == "ddp":
                # Do DDP backtracking line search here
                self.linesearch.ddp_backtracking_linesearch(self.qp_solver.hpipm_qp_solver,
                                                        self.nlp_problem,
                                                        self.log,
                                                        self.iterate,
                                                        self.direction)
            else:
                raise RuntimeError("Wrong mode given!")

            # Add statistics
            self.step_accepted = True
            self.n_iter += 1

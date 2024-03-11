"""
This function contains all the necessary input given to the solver
"""
# Import standard libraries
from __future__ import annotations # <-still need this.
from typing import TYPE_CHECKING
import casadi as cs
# Import self-written libraries
if TYPE_CHECKING:
    from .logger import Logger


class NLPProblem:

    def __init__(self, feasibility_problem_data: dict):
        
        self.number_variables = feasibility_problem_data['n_variables']
        self.number_constraints = feasibility_problem_data['feas_nlp_n_constraints']
        self.number_parameters = feasibility_problem_data['n_parameters']

        self.n_stages = feasibility_problem_data['N']
        self.n_states = feasibility_problem_data['n_states_per_stage']
        self.n_controls = feasibility_problem_data['n_controls_per_stage']
        self.discrete_dynamics = feasibility_problem_data['feas_nlp_discrete_dynamics_fun']
        self.initial_condition_indeces = feasibility_problem_data['feas_nlp_g_initial_condition_indeces']
        self.dynamics_constraints_indeces = feasibility_problem_data['feas_nlp_g_dynamics_indeces']

        self.f_function = feasibility_problem_data['feas_nlp_f_fun']
        self.gradient_f_function = feasibility_problem_data['feas_nlp_grad_f_fun']
        self.hessian_lagrangian_function = feasibility_problem_data['feas_nlp_hessian_fun']
        self.g_function = feasibility_problem_data['feas_nlp_g_fun']
        self.jacobian_g_function = feasibility_problem_data['feas_nlp_jac_g_fun']
        self.norm_control_gradient_fun = feasibility_problem_data['feas_nlp_control_obj_gradient_fun']
        self.obj_control_fun = feasibility_problem_data['feas_nlp_control_obj_fun']
        # gradient of Lagrangian can be calculated from other quantities

        self.lbg = feasibility_problem_data['feas_nlp_lbg'] # should be zero!
        self.ubg = feasibility_problem_data['feas_nlp_ubg']

        # Original Problem
        self.nlp_lbg = feasibility_problem_data['nlp_lbg'] # should be zero!
        self.nlp_ubg = feasibility_problem_data['nlp_ubg']
        self.nlp_g_function = feasibility_problem_data['nlp_g_fun']
        self.nlp_jacobian_g_function = feasibility_problem_data['nlp_jac_g_fun']
        self.nlp_initial_condition_indeces = feasibility_problem_data['nlp_g_initial_condition_indeces']
        self.nlp_dynamics_indeces = feasibility_problem_data['nlp_g_dynamics_indeces']
        self.nlp_n_constraints = feasibility_problem_data['nlp_n_constraints']
        self.nlp_g_lower_bounded_indeces = feasibility_problem_data['nlp_g_lower_bounded_indeces']
        self.nlp_g_lower_upper_bounded_indeces = feasibility_problem_data['nlp_g_lower_upper_bounded_indeces']
        self.nlp_g_upper_bounded_indeces = feasibility_problem_data['nlp_g_upper_bounded_indeces']
        self.nlp_x_control_positions = feasibility_problem_data['nlp_x_control_positions']

        # Feasible Initial Guess
        self.x0 = feasibility_problem_data['x0']

    ###########################################################################
    # Methods for function evaluations
    ###########################################################################

    def eval_f(self, x: cs.DM, p: cs.DM, log: Logger):
        """
        Evaluates the objective function. And stores the statistics of it.

        Args:
            x (Casadi DM vector): the value of the states where f is evaluated

        Returns:
            Casadi DM scalar: the value of f at the given x.
        """
        log.increment_n_eval_f()
        return self.f_function(x, p)

    def eval_g(self, x: cs.DM, p: cs.DM, log: Logger):
        """
        Evaluates the constraint function. And stores the statistics of it.

        Args:
            x (Casadi DM vector): the value of the states where g is evaluated

        Returns:
            _type_: _description_
        """
        log.increment_n_eval_g()
        return self.g_function(x, p)
    
    def eval_nlp_g(self, x: cs.DM, p: cs.DM, log: Logger):
        """
        Evaluates the constraint function. And stores the statistics of it.

        Args:
            x (Casadi DM vector): the value of the states where g is evaluated

        Returns:
            _type_: _description_
        """
        log.increment_n_eval_nlp_g()
        return self.nlp_g_function(x, p)

    def eval_gradient_f(self, x: cs.DM, p: cs.DM, log: Logger):
        """
        Evaluates the objective gradient function. And stores the statistics 
        of it.
        
        Args:
            x (Casadi DM vector): the value of the states where gradient of f 
            is evaluated

        Returns:
           Casadi DM vector: the value of g at the given x.
        """
        log.increment_n_eval_gradient_f()
        return self.gradient_f_function(x, p)

    def eval_jacobian_g(self, x: cs.DM, p: cs.DM, log: Logger):
        """
        Evaluates the constraint jacobian function. And stores the
        statistics of it.

        Args:
            x (Casadi DM vector): the value of the states where Jacobian of g 
            is evaluated

        Returns:
            Casadi DM vector: the value of g at the given x.
        """
        log.increment_n_eval_jacobian_g()
        return self.jacobian_g_function(x, p)
    
    def eval_nlp_jacobian_g(self, x: cs.DM, p: cs.DM, log: Logger):
        """
        Evaluates the constraint jacobian function. And stores the
        statistics of it.

        Args:
            x (Casadi DM vector): the value of the states where Jacobian of g 
            is evaluated

        Returns:
            Casadi DM vector: the value of g at the given x.
        """
        log.increment_n_eval_nlp_jacobian_g()
        return self.nlp_jac_g_function(x, p)

    def eval_gradient_lagrangian(self, 
                                   gradient_f: cs.DM,
                                   jacobian_g: cs.DM, 
                                   lam_g: cs.DM, 
                                   log:Logger):
        """
        Evaluates the gradient of the Lagrangian at x, lam_g, and lam_x.
        
        Args:
            x (Casadi DM vector): the value of the states where Jacobian of g 
            lam_g (Casadi DM vector): value of multipliers for constraints g
            lam_x (Casadi DM vector): value of multipliers for states x

        Returns:
            Casadi DM vector: the value of gradient of Lagrangian
        """
        log.increment_n_eval_gradient_lagrangian()
        return gradient_f + jacobian_g.T @ lam_g

    def eval_hessian_lagrangian(self,
                                x: cs.DM,
                                p: cs.DM,
                                log: Logger):
        """
        Evaluates the Hessian of Lagrangian. And stores the statistics 
        of it.
        """
        log.increment_n_eval_hessian_lagrangian()
        return self.hessian_lagrangian_function(x, p)

    ###########################################################################
    # Methods for interaction from  OCP to NLP and vice versa
    ###########################################################################
    
    def forward_dynamics_simulation(self, initial_state, controls):
        states = [initial_state]
        curr_state = initial_state
        for i in range(self.n_stages):
            curr_state = self.discrete_dynamics(curr_state, controls[i])
            states.append(curr_state)

        return states
    
    def disassemble_iterate(self, x: cs.DM):
        """
        Disassembles the whole iterate.
        """
        states = []
        controls = []
        counter = 0
        for i in range(self.n_stages+1):
            if i <self.n_stages:
                states.append(x[counter:counter+self.n_states])
                counter += self.n_states
                controls.append(x[counter:counter+self.n_controls])
                counter += self.n_controls
            else:
                states.append(x[counter:counter+self.n_states])

        return states, controls

    def assemble_iterate(self, states, controls):
        iterate = []
        for i in range(self.n_stages+1):
            if i <self.n_stages:
                iterate.append(states[i])
                iterate.append(controls[i])
            else:
                iterate.append(states[i])
        return cs.vertcat(*iterate)
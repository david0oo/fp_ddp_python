"""
This file gives some functions that creates the correct matrices to ship them 
to HPIPM.
"""
import casadi as cs
import numpy as np

class HPIPMConverter:

    def __init__(self, init_dict) -> None:
        
        self.N = init_dict['N']
        self.n_states_per_stage = init_dict['n_states_per_stage']
        self.n_controls_per_stage = init_dict['n_controls_per_stage']
        self.initial_condition_indeces = init_dict['feas_nlp_g_initial_condition_indeces']
        self.dynamics_indeces = init_dict['feas_nlp_g_dynamics_indeces']

    def disassemble_iterate(self, x: cs.DM):
        """
        Disassembles the whole iterate.
        """
        states = []
        controls = []
        counter = 0
        for i in range(self.N+1):
            if i <self.N:
                states.append(np.array(x[counter:counter+self.n_states_per_stage]))
                counter += self.n_states_per_stage
                controls.append(np.array(x[counter:counter+self.n_controls_per_stage]))
                counter += self.n_controls_per_stage
            else:
                states.append(np.array(x[counter:counter+self.n_states_per_stage]))

        return states, controls

    def assemble_iterate(self, states, controls):
        iterate = []
        for i in range(self.N+1):
            if i <self.N:
                iterate.append(states[i])
                iterate.append(controls[i])
            else:
                iterate.append(states[i])
        return cs.vertcat(*iterate)

    def transform_hessian_to_small_hessians(self, hessian: cs.DM):

        hessians_states = []
        hessians_controls = []
        hessians_mixed = []

        counter = 0
        for i in range(self.N+1):

            if i != self.N:
                nx = self.n_states_per_stage
                nu = self.n_controls_per_stage
                hessian_tmp = hessian[counter:counter+nx+nu, counter:counter+nx+nu]
                hessians_states.append(np.array(hessian_tmp[:nx, :nx]))
                hessians_controls.append(np.array(hessian_tmp[nx:nx+nu, nx:nx+nu]))
                hessians_mixed.append(np.array(hessian_tmp[nx:nx+nu, :nx])) # Correct??
            
                counter += int(nx + nu)
            else:
                nx = self.n_states_per_stage
                hessian_tmp = np.array(hessian[counter:counter+nx, counter:counter+nx])
                hessians_states.append(hessian_tmp)

        return hessians_states, hessians_controls, hessians_mixed

    def transform_gradient_to_stagewise_gradients(self, gradient: cs.DM):

        gradients_states = []
        gradients_controls = []

        counter = 0
        nx = self.n_states_per_stage
        nu = self.n_controls_per_stage
        for i in range(self.N+1):

            if i != self.N:
                gradient_tmp = gradient[counter:counter+nx+nu]
                gradients_states.append(np.array(gradient_tmp[:nx]))
                gradients_controls.append(np.array(gradient_tmp[nx:]))
                counter += int(nx + nu)
            else:
                gradient_tmp = gradient[counter:counter+nx]
                gradients_states.append(np.array(gradient_tmp))

        return gradients_states, gradients_controls
    
    def transform_dynamics_to_stagewise_dynamics(self,
                                                 current_bounds: cs.DM,
                                                 current_jacobian: cs.DM):

        dynamic_constraints = current_bounds
        dynamic_jacobian = current_jacobian

        A_matrices = []
        B_matrices = []
        b_vectors = []

        counter = 0
        nx = self.n_states_per_stage
        nu = self.n_controls_per_stage
        for i in range(self.N):

            b_vectors.append(np.array(dynamic_constraints[i*nx:i*nx+nx]))

            jac_tmp = dynamic_jacobian[i*nx:i*nx+nx, counter:counter+nx+nu]
            A_matrices.append(-np.array(jac_tmp[:, :nx]))
            B_matrices.append(-np.array(jac_tmp[:, nx:nx+nu]))
        
            counter += int(nx + nu)

        return A_matrices, B_matrices, b_vectors
    
"""
Chen-Allgoewer Example
===================

Problem is supposed to be unstable.
"""
import numpy as np
# Import the project
from rockit import *
import casadi as cs
from matplotlib import pyplot as plt
from scipy.linalg import solve_discrete_are
import matplotlib

def latexify():
    params = {
              'axes.labelsize': 10,
              'axes.titlesize': 10,
              'legend.fontsize': 10,
              'xtick.labelsize': 10,
              'ytick.labelsize': 10,
              'text.usetex': True,
              'font.family': 'serif'
              }

    matplotlib.rcParams.update(params)

latexify()
#%%
# Problem specification
# ---------------------

# Start an optimal control environment with a time horizon of 10 seconds
# starting from t0=0s.
def create_ocp():
    ocp = Ocp(t0=0, T=5)

    # Define two scalar states (vectors and matrices also supported)
    x1 = ocp.state()
    x2 = ocp.state()
    x = cs.vertcat(x1, x2)

    # Define one piecewise constant control input
    u = ocp.control()

    # Parameter
    mu = 0.7
    Q = np.array([[0.5, 0], [0, 0.5]])
    R = np.array([[0.8]])
    P = np.array([[10.0, 0], [0, 10.0]])

    # Specify differential equations for states
    ocp.set_der(x1, x2 + u*(mu + (1-mu)*x2))
    ocp.set_der(x2, x1 + u*(mu-4*(1-mu)*x2))

    # Lagrange objective term: signals in an integrand
    ocp.add_objective(0.5*ocp.integral(0.5*x.T @ Q @ x + 0.5*x.T @ R @ x))
    # Mayer objective term: signals evaluated at t_f = t0_+T
    ocp.add_objective(ocp.at_tf(0.5*x.T @ P @ x))

    # Path constraints
    #  (must be valid on the whole time domain running from `t0` to `tf`,
    #   grid options available such as `grid='inf'`)
    u_bar = 1.5
    ocp.subject_to(-u_bar <= (u <= u_bar ))
    
    # Boundary constraints
    ocp.subject_to(ocp.at_t0(x1) == 0.42)
    ocp.subject_to(ocp.at_t0(x2) == 0.45)

    # Add obstacle avoidance constraint
    ocp.subject_to(ocp.at_tf(x1) == 0.0)
    ocp.subject_to(ocp.at_tf(x2) == 0.03)
    # ocp.subject_to(ocp.at_tf(x1) == 0.0)
    # ocp.subject_to(ocp.at_tf(x2) == 0.0)
    # Pick an NLP solver backend
    #  (CasADi `nlpsol` plugin):
    ocp.solver('ipopt')
    # Pick a solution method
    N = 20
    method = MultipleShooting(N=N, M=10, intg='rk')
    ocp.method(method)

    # ------------------------------------
    # Create initial guess
    # ------------------------------------
    discrete_dynamics = ocp.discrete_system()
    x_sym = cs.MX.sym('x', 2)
    u_sym = cs.MX.sym('x', 1)
    A_fun = cs.Function("A_fun", [x_sym, u_sym], [cs.jacobian(discrete_dynamics(x0=x_sym, u=u_sym, T=0.25)['xf'], x_sym)])
    B_fun = cs.Function("B_fun", [x_sym, u_sym], [cs.jacobian(discrete_dynamics(x0=x_sym, u=u_sym, T=0.25)['xf'], u_sym)])

    A = np.array(A_fun(cs.vertcat(0,0), 0))
    B = np.array(B_fun(cs.vertcat(0,0), 0))
    Q = np.array([[0.5, 0], [0, 0.5]])
    R = np.array([[0.8]])
    K = compute_lqr_feedback_law(A, B, Q, R)
    

    x_curr = cs.vertcat(0.42, 0.45)
    X_init = x_curr
    U_init = []
    for k in range(N):
        u_curr = -K @ x_curr
        x_curr = discrete_dynamics(
            x0=x_curr, u=u_curr, T=0.25)['xf']
        X_init = cs.horzcat(X_init, x_curr)
        U_init = cs.horzcat(U_init, u_curr)

    # Set initial guesses for states, controls and variables.
    #  Debug forward simulation
    ocp.set_initial(x1, X_init[0,:])          # Function of time
    ocp.set_initial(x2, X_init[1,:])                 # Constant
    ocp.set_initial(u, U_init) # Matrix

    # Solve
    # try:
    #     sol = ocp.solve()
    # except:
    #     print("Here")
    #     sol = ocp.non_converged_solution

    return ocp

def get_control_sol(ocp: Ocp, sol: cs.DM):
        """
        Retrieves the controls from all decision variables.
        (Use after optimization)
        
        Args:
            sol (Casadi DM vector): solution vector of the OCP.

        Returns:
            Casadi DM vector: vector just containing the controls.
        """
        nx = ocp.nx
        nu = ocp.nu
        N = ocp._method.N
        control_sol = []
        ind_count = 0
        for k in range(N+1):
            ind_count += nx
            if k < N:
                control_sol.append(sol[ind_count:ind_count+nu])
                ind_count += nu
        # ind_count += 1
        control_sol = cs.reshape(cs.vertcat(*control_sol), nu, N)

        return control_sol

def get_state_sol(ocp: Ocp, sol: cs.DM):
        """
        Retrieves the states from all decision variables.
        (Use after optimization)

        Args:
            sol (Casadi DM vector): solution vector of the OCP.

        Returns:
            Casadi DM vector: vector just containing of the states.
        """
        nx = ocp.nx
        nu = ocp.nu
        N = ocp._method.N
        states_sol = []
        ind_count = 0
        for k in range(N+1):
            states_sol.append(sol[ind_count:ind_count+nx])
            ind_count += nx
            if k < N:
                ind_count += nu
        # ind_count += 1

        return cs.vertcat(*states_sol)

def get_particular_states(ocp: Ocp, sol: cs.DM):
        states_sol = get_state_sol(ocp, sol)
        x1_sol = states_sol[0::2]
        x2_sol = states_sol[1::2]
        return (x1_sol, x2_sol)

def compute_lqr_feedback_law(A, B, Q, R):
    P = solve_discrete_are(A, B, Q, R,)
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return K

def plot_trajectory(ocp: Ocp, list_sol:list, list_labels:list):

    plt.figure(figsize=(4.5, 2.9))
    for i in range(len(list_sol)):
        x1_opt, x2_opt = get_particular_states(ocp, list_sol[i])
        plt.plot(x1_opt, x2_opt, label=list_labels[i])
    plt.xlabel("$x_1$", fontsize=14)
    plt.ylabel("$x_2$", fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.xlim((-0.2, 0.5))
    plt.ylim((0.0, 0.6))
    # plt.title('Trajectory')
    # plt.savefig(f"feasible_trajectory_chen_allgoewer.pdf", dpi=300, bbox_inches='tight', pad_inches=0.01)
    plt.show()

def plot_controls(ocp: Ocp, list_sol:list, list_labels:list):

    plt.figure(figsize=(10, 4))
    time = np.linspace(0,5,20)
    plt.plot(time, -1.5*np.ones(20), color='r')
    plt.plot(time, 1.5*np.ones(20), color='r')
    for i in range(len(list_sol)):
        u = get_control_sol(ocp, list_sol[i])
        plt.plot(time, np.array(u).squeeze(), label=list_labels[i])
    plt.xlabel("time", fontsize=14)
    plt.ylabel("$u$", fontsize=14)
    plt.tight_layout()
    plt.grid(True)
    plt.legend()
    plt.title('Control over time')
    plt.show()

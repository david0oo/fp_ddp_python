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
import matplotlib.patches as patches
import matplotlib as mpl
from matplotlib.animation import FuncAnimation

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

    mpl.rcParams.update(params)

latexify()
#%%
# Problem specification
# ---------------------
class CartPendulumVisualization:
    def __init__(self, moon_x=0.5, L=0.8):
        self.L = L
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim([-1.0, 7.0])
        self.ax.set_ylim([-1.5, 1.5])
        self.ax.set_aspect('equal')
        # self.ax.grid()
        self.circle = patches.Circle((moon_x, 0.9), 0.3,color='r')
        self.line = patches.Rectangle((-1.0, -0.005), 8, 0.01, fc='gray')
        self.cart = patches.Rectangle((-0.1, -0.1), 0.2, 0.2, fc='r')
        self.pendulum = patches.Rectangle((-0.02, 0), 0.04, L, fc='b')
        self.ax.add_patch(self.circle)
        self.ax.add_patch(self.line)
        self.ax.add_patch(self.cart)
        self.ax.add_patch(self.pendulum)

    def visualize(self, x, theta):
        self.cart.set_xy([x-0.1, -0.1])
        # rotate the pendulum over an angle theta
        # self.pendulum.set_transform(mpl.transforms.Affine2D().rotate_around(
        #     0.00, 0., (-np.pi + theta)).translate(x, 0) + self.ax.transData)
        self.pendulum.set_transform(mpl.transforms.Affine2D().rotate_around(
            0.00, 0., (theta)).translate(x, 0) + self.ax.transData)

        # set the position of the pendulum
        self.fig.canvas.draw()
        # self.fig.canvas.flush_events()

class VisualizationFunctor:
    def __init__(self, vis, x, theta, L=1.):
        self.x = x
        self.theta = theta
        self.vis: CartPendulumVisualization = vis

    def __call__(self, frame):
        self.vis.visualize(self.x[frame], self.theta[frame])

# Start an optimal control environment with a time horizon of 10 seconds
# starting from t0=0s.
# def create_ocp(moon_x=0.7, initial_p_guess=None):
def create_ocp():

    ocp = Ocp(t0=0, T=1)
    # tmp_ocp = Ocp(t0=0, T=5)

    # Parameters
    M = 1.0  # cart mass [kg]
    m = 0.1       # pendulum mass [kg]
    l = 0.8       # pendulum length [m]
    g = 9.81      # gravitation [m/s^2]

    # max_f = 10.
    max_f = 20.
    # max_v = 8.
    max_v = 10.
    max_theta = cs.pi/4

    moon_x = ocp.parameter()
    moon_midpoint = cs.vertcat(moon_x, 0.9) # works
    moon_radius = 0.3

    p_end = 5.0

    # Define the states and controls 
    T = ocp.state()
    p = ocp.state()
    p_dot = ocp.state()
    theta = ocp.state()
    theta_dot = ocp.state()
    F = ocp.control()

    # Define some expressions
    denominator = M + m - m*cs.cos(theta)*cs.cos(theta)
    p_ddot = (-m*l*cs.sin(theta)*theta_dot*theta_dot + m*g*cs.cos(theta)*cs.sin(theta)+F)/denominator
    theta_ddot = (-m*l*cs.cos(theta)*cs.sin(theta)*theta_dot*theta_dot + F*cs.cos(theta)+(M+m)*g*cs.sin(theta))/(l*denominator)
    pendulum_position = cs.vertcat(p-l*cs.sin(theta),l*cs.cos(theta))

    # Define free-time derivatives
    ocp.set_der(T, 0.)
    ocp.set_der(p, p_dot * T)
    ocp.set_der(p_dot, p_ddot * T)
    ocp.set_der(theta, theta_dot *T)
    ocp.set_der(theta_dot, theta_ddot * T)

    # P2P motion initial constraints
    ocp.subject_to(ocp.at_t0(p == 0))
    ocp.subject_to(ocp.at_t0(p_dot == 0))
    ocp.subject_to(ocp.at_t0(theta == 0))
    ocp.subject_to(ocp.at_t0(theta_dot == 0))

    # Define constraints on actuatorconstraints
    ocp.subject_to(ocp.at_t0(T) >= 1e-3)
    ocp.subject_to(ocp.at_t0(T) <= 10.0)
    ocp.subject_to(-max_f <= (F <= max_f), include_last=False)
    ocp.subject_to(-1.0 <= (p <= 7.0), include_first=False, include_last=False)
    ocp.subject_to(-max_v <= (p_dot <= max_v), include_first=False, include_last=False)
    ocp.subject_to(-max_theta <= (theta <= max_theta), include_first=False, include_last=False)
    
    # Obstacle avoidance constraint
    ocp.subject_to(cs.sumsqr((pendulum_position - moon_midpoint)) >= moon_radius**2, include_first=False, include_last=False)

    # P2P motion terminal constraints
    ocp.subject_to(ocp.at_tf(p == p_end))
    ocp.subject_to(ocp.at_tf(theta == 0))
    ocp.subject_to(ocp.at_tf(p_dot == 0))
    ocp.subject_to(ocp.at_tf(theta_dot == 0))

    # Minimal time
    # ocp.add_objective(ocp.at_tf(T) + 1e-5*ocp.integral(F**2))
    ocp.add_objective(ocp.at_tf(T)) #+ 1e-5*ocp.integral(F**2))

    # Pick an NLP solver backend
    #  (CasADi `nlpsol` plugin):
    ocp.solver('ipopt')
    # Pick a solution method
    N = 100
    # method = MultipleShooting(N=N, M=10, intg='rk')
    method = MultipleShooting(N=N, intg='rk')
    ocp.method(method)

    # # ------------------------------------
    # # Create initial guess
    # # ------------------------------------
    # # Set initial guesses for states, controls and variables.
    # Load the previous solution
    # T_sol = cs.DM.from_file("no_obstacle_time_solution.mtx")
    # p_sol = cs.DM.from_file("no_obstacle_p_solution.mtx")
    # p_dot_sol = cs.DM.from_file("no_obstacle_p_dot_solution.mtx")
    # theta_sol = cs.DM.from_file("no_obstacle_theta_solution.mtx")
    # theta_dot_sol = cs.DM.from_file("no_obstacle_theta_dot_solution.mtx")
    # F_sol = cs.DM.from_file("no_obstacle_F_solution.mtx")

    # T_sol = cs.DM.from_file("obstacle_time_solution.mtx")
    # p_sol = cs.DM.from_file("obstacle_p_solution.mtx")
    # p_dot_sol = cs.DM.from_file("obstacle_p_dot_solution.mtx")
    # theta_sol = cs.DM.from_file("obstacle_theta_solution.mtx")
    # theta_dot_sol = cs.DM.from_file("obstacle_theta_dot_solution.mtx")
    # F_sol = cs.DM.from_file("obstacle_F_solution.mtx")

    T_sol = 5.0
    # p_sol = cs.vertcat(cs.linspace(0, -1.0, 10), cs.linspace(-1.0, 6.0, 80), cs.linspace(6.0, 5.0, 11)) # for p_end = 5.0
    p_sol = cs.vertcat(cs.linspace(0, -1.0, 10), cs.linspace(-1.0, p_end+1.0, 80), cs.linspace(p_end+1, p_end, 11))
    # theta_sol = cs.vertcat(cs.linspace(0, cs.pi/6, 10), cs.linspace(cs.pi/6, -cs.pi/6, 80), cs.linspace(-cs.pi/6, 0, 11))

    # #  Debug forward simulation
    # ocp.set_initial(T, 5.0)          # Function of time
    ocp.set_initial(T, T_sol)          # Function of time
    ocp.set_value(moon_x, 0.7)
    # if initial_p_guess is not None:
    #      ocp.set_initial(p, initial_p_guess)          # Function of time
    # else:
    #      ocp.set_initial(p, p_sol)          # Function of time
         
    # ocp.set_initial(p_dot, p_dot_sol)          # Function of time
    # ocp.set_initial(theta, theta_sol)          # Function of time
    # ocp.set_initial(theta_dot, theta_dot_sol)          # Function of time
    # ocp.set_initial(F, F_sol[:-1])          # Function of time
    # ocp.set_initial(F, F_sol)          # Function of time

    ocp._transcribe()
    # Solve
    # try:
    #     sol = ocp.solve()
    # except:
    #     print("Here")
    #     sol = ocp.non_converged_solution

    # _, p_sol = sol.sample(p, grid='control')
    # _, p_dot_sol = sol.sample(p_dot, grid='control')
    # _, theta_sol = sol.sample(theta, grid='control')
    # _, theta_dot_sol = sol.sample(theta_dot, grid='control')
    # _, time = sol.sample(T, grid='control')
    # _, F_sol = sol.sample(F, grid='control')

    # cs.DM(p_sol).to_file("no_obstacle_p_solution.mtx")
    # cs.DM(p_dot_sol).to_file("no_obstacle_p_dot_solution.mtx")
    # cs.DM(theta_sol).to_file("no_obstacle_theta_solution.mtx")
    # cs.DM(theta_dot_sol).to_file("no_obstacle_theta_dot_solution.mtx")
    # cs.DM(time).to_file("no_obstacle_time_solution.mtx")
    # cs.DM(F_sol).to_file("no_obstacle_F_solution.mtx")

    # cs.DM(p_sol).to_file("obstacle_p_solution.mtx")
    # cs.DM(p_dot_sol).to_file("obstacle_p_dot_solution.mtx")
    # cs.DM(theta_sol).to_file("obstacle_theta_solution.mtx")
    # cs.DM(theta_dot_sol).to_file("obstacle_theta_dot_solution.mtx")
    # cs.DM(time).to_file("obstacle_time_solution.mtx")
    # cs.DM(F_sol).to_file("obstacle_F_solution.mtx")


    # print("p solution: ", p)
    # tsa, theta_sol = sol.sample(theta, grid='control')
    # print("theta solution: ", theta)

    # tsa, pendulum_position = sol.sample(pendulum_position, grid='control')
    # x_pos = pendulum_position[:,0]
    # y_pos = pendulum_position[:,1]

    # plt.figure(figsize=(10, 4))
    # plt.plot(x_pos, y_pos, label="Pole position")
    # plt.xlabel("$x_1$", fontsize=14)
    # plt.ylabel("$x_2$", fontsize=14)
    # plt.grid(True)
    # plt.legend()
    # plt.title('Trajectory')
    # plt.show()

    # print("Optimal time: ", time[0])
    # # Plot animation
    # fps = int(N/time[0])
    # res_x = np.array(p_sol).squeeze()
    # res_theta = np.array(theta_sol).squeeze()
    # vis = CartPendulumVisualization()
    # ani = FuncAnimation(
    #     vis.fig, VisualizationFunctor(vis, res_x, res_theta),
    #     frames=range(N),
    #     interval=34)
    # plt.show(block=True)

    # ocp._untranscribe()
    # ocp.set_initial(T, 4.0)          # Function of time
    # ocp.set_initial(p, p_sol)
    # ocp._transcribe()

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
        x1_sol = states_sol[0::5]
        x2_sol = states_sol[1::5]
        x3_sol = states_sol[2::5]
        x4_sol = states_sol[3::5]
        x5_sol = states_sol[4::5]
        return (x1_sol, x2_sol)

def plot_trajectory(ocp: Ocp, list_sol:list, list_labels:list):

    plt.figure(figsize=(10, 4))
    for i in range(len(list_sol)):
        x1_opt, x2_opt = get_particular_states(ocp, list_sol[i])
        plt.plot(x1_opt, x2_opt, label=list_labels[i])
    plt.xlabel("$x_1$", fontsize=14)
    plt.ylabel("$x_2$", fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.title('Trajectory')
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
    plt.grid(True)
    plt.legend()
    plt.title('Control over time')
    plt.show()


if __name__ == "__main__":
    create_ocp()
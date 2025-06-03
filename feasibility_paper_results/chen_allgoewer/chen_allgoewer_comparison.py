# Include standard
import casadi as cs
import sys
import os
print(os.path.abspath(__file__))
dir_name = os.path.dirname(os.path.abspath(__file__))
include_dir = os.path.join(dir_name, "..", "..", "test_problems")
sys.path.append(include_dir)

from chen_allgoewer_unstable import create_ocp, plot_trajectory, plot_controls

from fp_ddp_python.solver import FeasibilityProblemSolver
from fp_ddp_python.ocp_transformer import OCP_To_Data_Transformer
from fp_ddp_python.plotting import Plotter

# Create the OCP problem
ocp = create_ocp()
transformer = OCP_To_Data_Transformer()
feasibility_problem_data = transformer.transform(ocp,smoothmax=False)

opts = {}
opts['objective_tol'] = 1e-15
# Create DMS solver
solver_dms = FeasibilityProblemSolver(feasibility_problem_data, mode="dms", init_feasible=False)
solver_dms.iterate.set_penalty(1.0)
solver_dms.solve(feasibility_problem_data)

solver_dms2 = FeasibilityProblemSolver(feasibility_problem_data, mode="dms", init_feasible=False)
solver_dms2.iterate.set_penalty(0.1)
solver_dms2.solve(feasibility_problem_data)

solver_dms3 = FeasibilityProblemSolver(feasibility_problem_data, mode="dms", init_feasible=False)
solver_dms3.iterate.set_penalty(0.01)
solver_dms3.solve(feasibility_problem_data)

# Create DSS solver
solver_dss = FeasibilityProblemSolver(feasibility_problem_data, mode="dss", init_feasible=True, opts=opts)
solver_dss.solve(feasibility_problem_data)

# Create DDP solver
solver_ddp = FeasibilityProblemSolver(feasibility_problem_data, mode="ddp", init_feasible=True)
solver_ddp.solve(feasibility_problem_data)

plot_trajectory(ocp, [ feasibility_problem_data['x0'], solver_dms.iterate.x_k, solver_dss.iterate.x_k, solver_ddp.iterate.x_k], ["Initial Guess", "DMS $\\sigma = 1.0$", "DSS", "DDP"])
plot_controls(ocp, [ feasibility_problem_data['x0'], solver_dms.iterate.x_k, solver_dss.iterate.x_k, solver_ddp.iterate.x_k], ["Initial Guess", "DMS", "DSS", "DDP"])

plotter = Plotter()

labels = ["DMS $\\sigma = 1.0$", "DMS $\\sigma = 0.1$", "DMS $\\sigma = 0.01$", "DSS", "FP-DDP"]
linestyles = ["solid", "dashdot", "dotted", "dashed", "solid"]
color = ['gray', 'k','k','k','k']
data = [solver_dms.log.kkt_residual, solver_dms2.log.kkt_residual, solver_dms3.log.kkt_residual,solver_dss.log.kkt_residual, solver_ddp.log.kkt_residual]
plotter.plot_convergence_rate(data, labels, color=color, linestyles=linestyles, save_name="KKT_residual_chen_allgoewer")


# FP-DDP
A prototypical implementation of a DDP solver for finding feasible trajectories in discrete-time optimal control. A high-performance implementation is available within [acados](https://github.com/acados/acados).

This repository includes the code to check and reproduce the simulation results of the paper ["Fast Generation of Feasible Trajectories in Direct Optimal Control"](https://arxiv.org/abs/2403.10115).

## Installation on Linux

In order to use the solver you need to install the following dependencies:
- The protoypical solver is written in `python3`. Therefore, get a recent `python3` version
- Install packages.
```bash
pip install casadi rockit-meco matplotlib numpy scipy
```
- Get `HPIPM` in python. Since we did some small changes to the python HPIPM interface, use the following repo and branch: <a href="https://github.com/sandmaennchen/hpipm/tree/nice_interface">new HPIPM python interface</a>
- Install the FP-DDP solver: Navigate to the fp_ddp_python directory and do
```bash
pip install -e .
```
For generating creating the acados results. Please install acados according to its installation instructions.

## Run the experiments
- Navigate to the folder `feasibility_paper_results`

### Example 1: Fixed Time, Unstable OCP: Chen-Allgoewer Problem
- Run the file `chen_allgoewer_comparison.py`

### Example 2: Free-Time Cart Pendulum with obstacle avoidance
- For reproducing the plot: Run the file `plot_timings_performance_plot.py`
- For running the experiments, please run the following files individually:
  - For FP-DDP within acados run: `acados_cart_pendulum/pendulum_p2p_obstacle.py`
  - For IPOPT with code generation run: `cart_pendulum/ipopt_code_generation/cart_pendulum_comparison.py`
  - For IPOPT without code generation run: `cart_pendulum/ipopt_no_codegen/cart_pendulum_comparison.py`
  - For Scipy run: `cart_pendulum/cart_pendulum_comparison.py`

## Citing
```
@article{Kiessling2024,
	publisher = {IEEE},
	year = {2024},
	journal = {IEEE Control Systems Letters},
	author = {Kiessling, David and Baumg{\"a}rtner, Katrin and Frey, Jonathan and Decr{\'e}, Wilm and Swevers, Jan and Diehl, Moritz},
	title = {Fast Generation of Feasible Trajectories in Direct Optimal Control},
}
```

## Contact
For questions please send an email to david.kiessling@kuleuven.be

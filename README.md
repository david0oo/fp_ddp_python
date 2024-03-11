# ffp_ddp
A prototypical implementation of a DDP solver for finding feasible trajectories in discrete-time optimal control

This repository includes a prototypical python implementation that makes it possible to check the simulation results of the paper "Fast Generation of Feasible Trajectories in Direct Optimal Control".

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

## Run the experiments
- Navigate to the folder `feasibility_paper_results`
- Run the Chen Allgoewer Example: Run the file `chen_allgoewer_comparison.py`
- Run the cart pendulum example: Run the file `cart_pendulum_comparison.py` to get the simulation results and `plot_cart_pendulum_results.py` to get the performance plots

## Contact
For questions please send an email to david.kiessling@kuleuven.be

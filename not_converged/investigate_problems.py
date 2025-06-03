import re
from pathlib import Path
from collections import OrderedDict
# import leap_c.examples.pendulum_on_a_cart  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
# import sac_fop_const  # noqa: F401
# import sac_fop_live_il  # noqa: F401
import seaborn as sns
from acados_template.acados_ocp_solver import AcadosOcpSolver
# from registry import create_task
from mpc_pendulum import PendulumOnCartMPC

PARAMS_SWINGUP = OrderedDict(
    [
        ("M", np.array([1.0])),  # mass of the cart [kg]
        ("m", np.array([0.1])),  # mass of the ball [kg]
        ("g", np.array([9.81])),  # gravity constant [m/s^2]
        ("l", np.array([0.8])),  # length of the rod [m]
        # The quadratic cost matrix is calculated according to L@L.T
        ("L11", np.array([np.sqrt(2e3)])),
        ("L22", np.array([np.sqrt(2e3)])),
        ("L33", np.array([np.sqrt(1e-2)])),
        ("L44", np.array([np.sqrt(1e-2)])),
        ("L55", np.array([np.sqrt(2e-1)])),
        ("Lloweroffdiag", np.array([0.0] * (4 + 3 + 2 + 1))),
        (
            "c1",
            np.array([0.0]),
        ),  # position linear cost, only used for non-LS (!) cost
        (
            "c2",
            np.array([0.0]),
        ),  # theta linear cost, only used for non-LS (!) cost
        (
            "c3",
            np.array([0.0]),
        ),  # v linear cost, only used for non-LS (!) cost
        (
            "c4",
            np.array([0.0]),
        ),  # thetadot linear cost, only used for non-LS (!) cost
        (
            "c5",
            np.array([0.0]),
        ),  # u linear cost, only used for non-LS (!) cost
        (
            "xref1",
            np.array([0.0]),
        ),  # reference position, only used for LS cost
        (
            "xref2",
            np.array([0.0]),
        ),  # reference theta, only used for LS cost
        (
            "xref3",
            np.array([0.0]),
        ),  # reference v, only used for LS cost
        (
            "xref4",
            np.array([0.0]),
        ),  # reference thetadot, only used for LS cost
        (
            "uref",
            np.array([0.0]),
        ),  # reference u, only used for LS cost
    ]
)


def extract_status_and_index(filename):
    match = re.search(r"status_(\d+)", filename)
    if match:
        status = int(match.group(1))
    else:
        raise Exception("it should always match")

    # Extract index (assuming it's a number after "status")
    index_match = re.search(r"\d+_(\d+)_", filename)
    index = int(index_match.group(1)) if index_match else float("inf")

    return status, index


def group_files_by_status(files_sorted):
    grouped_files = {1: [], 2: [], 4: []}

    for file in files_sorted:
        status, index = extract_status_and_index(file.name)
        if status in grouped_files:
            grouped_files[status].append(file)

    return grouped_files


def run_problem_instance(instance_status, instance_index_ls, files_grouped):
    instance = files_grouped[instance_status][
        3 * instance_index_ls : 3 * instance_index_ls + 3
    ]
    for path in instance:
        if "iterate" in path.name:
            instance_iterate_path = path
        elif "param" in path.name:
            instance_param_path = path
        elif "x0" in path.name:
            instance_x0_path = path
        else:
            raise Exception("Shouldn't happen")
    x0 = np.load(instance_x0_path)
    param = np.load(instance_param_path)
    print("X0: ", x0)
    print("Param: ", param)
    ocp_solver.load_iterate(instance_iterate_path)
    ocp_solver.set_p_global_and_precompute_dependencies(param.astype(np.float64))
    ocp_solver.solve_for_x0(
        x0.astype(np.float64),
        fail_on_nonzero_status=False,
        print_stats_on_failure=False,
    )
    print(
        ocp_solver.get_status(),
        ocp_solver.get_residuals(),
        ocp_solver.get_stats("nlp_iter"),
    )
    # ocp_solver.print_statistics()


# def load_and_plot_all(files_grouped, identifier: str):
#     for status in files_grouped.keys():
#         data_list = []
#         for i in range(len(files_grouped[status])):
#             data_path = files_grouped[status][i]
#             if identifier in data_path.name:
#                 data = np.load(data_path)
#                 data_list.append(data)
#         if len(data_list) == 0:
#             print("No examples found for status ", status)
#             continue

#         data_array = np.array(data_list)
#         assert data_array.ndim == 2
#         num_dimensions = data_array.shape[1]

#         if num_dimensions == 1:
#             fig, axes = plt.subplots(1, 1, figsize=(12, 6))
#             fig.suptitle(f"Histograms for Status {status}", fontsize=16)

#             sns.histplot(data_array[:, 0], kde=False, ax=axes, color="blue")  # type:ignore
#             axes.set_title("Histogram for Dimension 0")  # type:ignore
#             axes.set_xlabel("Value")  # type:ignore
#             axes.set_ylabel("Frequency")  # type:ignore

#             plt.tight_layout()
#         else:
#             fig, axes = plt.subplots(num_dimensions, 1, figsize=(12, 6))
#             fig.suptitle(f"Histograms for Status {status}", fontsize=16)

#             for dim in range(num_dimensions):
#                 sns.histplot(data_array[:, dim], kde=False, ax=axes[dim], color="blue")  # type:ignore
#                 axes[dim].set_title(f"Histogram for Dimension {dim}")  # type:ignore
#                 axes[dim].set_xlabel("Value")  # type:ignore
#                 axes[dim].set_ylabel("Frequency")  # type:ignore

#             plt.tight_layout()
#         plots = Path.cwd() / "plots"
#         plots.mkdir(exist_ok=True)
#         plt.savefig(
#             plots / f"Histograms_for_Status_{status}_identifier_'{identifier}'.png"
#         )


if __name__ == "__main__":
    task_name = "pendulum_balance"
    deterministic = True
    # task = create_task(task_name)

    load_path = Path.cwd()
    # NOTE: Don't forget to change N and T in the task.py accordingly when loading other problems
    # load_path = (
    #     load_path / "not_converged_problems_CARTPOLE_BALANCE_N=20_T=1_ZEROS_GLOB"
    # )
    load_path = (
        load_path / "non_converging_iterates"
    )
    # ocp_solver: AcadosOcpSolver = task.mpc.mpc.ocp_solver  # type:ignore
    cart_on_pole = PendulumOnCartMPC(
        N_horizon=20,
        T_horizon=1,
        learnable_params=["xref2"],
        params=PARAMS_SWINGUP,  # type: ignore
    )
    ocp = cart_on_pole.ocp
    ocp_solver = AcadosOcpSolver(ocp)

    files = list(load_path.iterdir())
    print("Number of non convergences in directory: ", len(files) / 3)
    files_grouped = group_files_by_status(files)
    for group in files_grouped.keys():
        files_grouped[group] = sorted(
            files_grouped[group], key=lambda f: extract_status_and_index(f.name)
        )

    assert len(files) == len(files_grouped[1]) + len(files_grouped[2]) + len(
        files_grouped[4]
    ), "Some files have been lost?"

    # instance_status = 4
    instance_status = 2

    indices = [1]#, 211, 469]

    for i in indices:
    # for i in range(3):
        run_problem_instance(
            instance_status=instance_status,
            instance_index_ls=i,
            files_grouped=files_grouped,
        )
    # load_and_plot_all(
    #     files_grouped=files_grouped,
    #     identifier="x0",
    # )
    # load_and_plot_all(
    #     files_grouped=files_grouped,
    #     identifier="param",
    # )

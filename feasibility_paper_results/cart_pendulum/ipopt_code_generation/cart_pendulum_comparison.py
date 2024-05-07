# Include standard 
import casadi as cs
import rockit as roc
import numpy as np
import sys
import os
import subprocess
import re
print(os.path.abspath(__file__))
dir_name = os.path.dirname(os.path.abspath(__file__))
import pickle
include_dir = os.path.join(dir_name, "..", "..","test_problems")
sys.path.append(include_dir)

from parametric_cartpendulum_time_optimal_obstacle import create_ocp
from fp_ddp_python.ocp_transformer import OCP_To_Data_Transformer


###############################################################################
# IPOPT
###############################################################################
def create_ipopt_executable_and_input_files(ocp: roc.Ocp):

    transformer = OCP_To_Data_Transformer()
    feasibility_problem_data = transformer.transform(ocp,smoothmax=False)

    obj = feasibility_problem_data['feas_nlp_f']
    g = feasibility_problem_data['feas_nlp_g']
    x = ocp._method.opti.x
    p = ocp._method.opti.p
    lbg = feasibility_problem_data['feas_nlp_lbg']
    ubg = feasibility_problem_data['feas_nlp_ubg']

    x0_vector = ocp.initial_value(ocp._method.opti.x)
    p0_vector = ocp.initial_value(ocp._method.opti.p)
    print("p0: ", p0_vector)

    nlp = {'x':x, 'p':p, 'f':obj, 'g':g}
    solver = cs.nlpsol('solver', 'ipopt', nlp)
    # res = solver(x0=x0_vector, p=p0_vector, lbg=lbg, ubg=ubg)

    kwargs = dict(x0=x0_vector, p=p0_vector, ubg = ubg, lbg = lbg)
    solver.generate('ipopt_solver.c',{"main":True})
    solver.generate_in("f_in.txt",solver.convert_in(kwargs))
    args = ["gcc","ipopt_solver.c", "-lipopt","-I" +cs.GlobalOptions.getCasadiIncludePath(),"-L"+cs.GlobalOptions.getCasadiPath(),"-lm","-o","ipopt_solver"]

    n_problems = 100
    moon_x = np.linspace(0.7, 4.3, n_problems)
    for i in range(n_problems):
        kwargs = dict(x0=x0_vector, p=cs.vertcat(moon_x[i]), ubg = ubg, lbg = lbg)
        solver.generate_in(dir_name + "/input_files/f_in"+str(i)+".txt",solver.convert_in(kwargs))

    # Compile the code
    # print(args)
    subprocess.Popen(args)
    

def run_ipopt_executable():
    time_codegen = []
    for i in range(100):
        times = []
        for j in range(20):
            command = "./ipopt_solver solver <" +dir_name + "/input_files/f_in"+str(i)+".txt > ipopt_output.txt"
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, cwd=dir_name)
            _, _ = process.communicate()
            # print(output)
            with open(dir_name +"/ipopt_output.txt") as f:
                content = f.read()
                print(content)
                result = re.search("Total seconds in IPOPT[ \t]+=[ \t]+([0-9]+\\.[0-9]+)", content)
                times.append(float(result.group(1)))
            f.close()
        time_codegen.append(min(times))
    print("Time in Codegen IPOPT: ", time_codegen)
    with open('ipopt_codegen_timings.pkl', 'wb') as f:
        pickle.dump(time_codegen, f)

###############################################################################
# Test the problems
###############################################################################
# def test_on_cart_pendulum(ddp=True, scipy=False, ipopt=False):
    
#     # Prepare lists for data
#     moon_x = cs.linspace(0.7, 4.3, 100)
#     ddp_list_n_eval_f = []
#     ddp_list_n_eval_g = []
#     ddp_list_n_eval_gradient_f = []
#     ddp_list_n_eval_jacobian_g = []
#     ddp_list_n_eval_gn_hessian = []

#     scipy_list_f_values = []
#     scipy_list_n_eval_f = []
#     scipy_list_n_eval_gradient_f = []
#     scipy_list_n_eval_hess_f = []

#     ipopt_list_n_eval_f = []
#     ipopt_list_n_eval_g = []
#     ipopt_list_n_eval_gradient_f = []
#     ipopt_list_n_eval_jacobian_g = []
#     ipopt_list_n_eval_hessian_l = []

#     # Run the simulation
#     for i in range(1):

#         # initial_p_guess = cs.DM.from_file(dir_name+"/initial_guess_ipopt_sol/ipopt_position_solution_at_"+str(i)+".mtx")
#         ocp = create_ocp(moon_x=moon_x[i])

#         # n_eval_f, n_eval_g, n_eval_gradient_f, n_eval_jac_g, n_eval_hess_l = test_ipopt_feasibility_problem(ocp)
#         ipopt_list_n_eval_f.append(n_eval_f)
#         ipopt_list_n_eval_g.append(n_eval_g)
#         ipopt_list_n_eval_gradient_f.append(n_eval_gradient_f)
#         ipopt_list_n_eval_jacobian_g.append(n_eval_jac_g)
#         ipopt_list_n_eval_hessian_l.append(n_eval_hess_l)

#     if ipopt:
#         with open('ipopt_results/ipopt_n_eval_f.pkl', 'wb') as f:
#             pickle.dump(ipopt_list_n_eval_f, f)
#         with open('ipopt_results/ipopt_n_eval_g.pkl', 'wb') as f:
#             pickle.dump(ipopt_list_n_eval_g, f)
#         with open('ipopt_results/ipopt_n_eval_gradient_f.pkl', 'wb') as f:
#             pickle.dump(ipopt_list_n_eval_gradient_f, f)
#         with open('ipopt_results/ipopt_n_eval_jacobian_g.pkl', 'wb') as f:
#             pickle.dump(ipopt_list_n_eval_jacobian_g, f)
#         with open('ipopt_results/ipopt_n_eval_hessian_l.pkl', 'wb') as f:
#             pickle.dump(ipopt_list_n_eval_hessian_l, f)

###############################################################################
# Run the simulation
###############################################################################
if __name__ == "__main__":
    # ocp = create_ocp()
    # create_ipopt_executable_and_input_files(ocp)
    run_ipopt_executable()
    # test_on_cart_pendulum(ddp=True, scipy=False, ipopt=False)

from collections import defaultdict
import numpy as np
import cython_solver
import python_solvers
import opencl_solver
import time
import pandas as pd
import matplotlib.pyplot as plt

from dataclasses import dataclass

@dataclass
class InitData:
    dt : float
    n_iters : int
    G : float = 6.6743 * 10 ** -11

@dataclass
class TestData:
    r_0 : np.ndarray
    v_0 : np.ndarray
    m_0 : np.ndarray

def bench(n_calls, func, *args, **kwargs):
    result = []
    for _ in range(n_calls):
        start = time.time()
        func(*args, **kwargs)
        result.append(time.time() - start)
    return np.mean(result)

def make_solver(solver_cls, init_data):
    return solver_cls(**init_data.__dict__)

def make_test_case(n_obj, n_dim):
    r = np.random.random_sample((n_obj, n_dim)).astype(np.float64)
    v = np.random.random_sample((n_obj, n_dim)).astype(np.float64)
    m = np.random.random_sample((n_obj,)).astype(np.float64)
    return TestData(r, v, m)

def plot_boosts(plot_data, k):
    for solver_name in plot_data:
        plt.plot(k, plot_data[solver_name], label=solver_name)
    plt.title("Run time")
    plt.xlabel("n bodies")
    plt.ylabel("time, s")
    plt.legend()
    plt.savefig("plots\\runtimes.png")

    plt.clf()
    for solver_name in plot_data:
        plt.plot(k, np.array(plot_data["python"]) / np.array(plot_data[solver_name]), label=solver_name)
    plt.title("Boost coeff")
    plt.ylabel("coefficient")
    plt.xlabel("n bodies")
    plt.legend()
    plt.savefig("plots\\boosts.png")


def main():
    solver_classes = {
        "odeint" : python_solvers.OdeintSolver,
        "python" : python_solvers.PythonVerletSolver,
        "cython" : cython_solver.CythonSolver,
        "multiprocessing" : python_solvers.MultiprocessingVerletSolver,
        "opencl" : opencl_solver.OpenCLSolver
    }

    n_dim = 2
    test_cases = [50, 100, 200]
    dt = pow(10, 7) * 1.5
    init_data_100 = InitData(dt, 100)

    results = {}
    for k in test_cases:
        test_case = make_test_case(k, n_dim)
        for solver_name in solver_classes:
            solver = make_solver(solver_classes[solver_name], init_data_100)
            results[(solver_name, k)] = bench(3, solver.solve, **test_case.__dict__)
            print(f"{solver_name} : {k} : {results[(solver_name, k)]}")

    df_data = [(k[0], k[1], results[k]) for k in results]
    df = pd.DataFrame(df_data, columns=["method", "n_obj", "time"])

    plot_data = defaultdict(list)
    for solver_name in solver_classes:
        for k in sorted(test_cases):
            plot_data[solver_name].append(results[(solver_name, k)])
    plot_boosts(plot_data, test_cases)

if __name__ == "__main__":
    main()
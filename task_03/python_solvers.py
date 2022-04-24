import functools
import numpy as np
from scipy.integrate import odeint
from dataclasses import dataclass
import multiprocessing as mp

@dataclass
class PythonSolver:
    dt : float
    n_iters : int
    G : float = 6.6743 * 10 ** -11

    @staticmethod
    def acceleration(r_i, m_i ,r_j, m_j, G=6.6743 * 10 ** -11):
        r_ij = r_j - r_i[:, None]
        f_m_denom = np.linalg.norm(r_ij, axis=2)[...,None] ** 3
        f_m = G * m_j[:, None] * np.divide(r_ij, f_m_denom, out=np.zeros_like(r_ij), where=f_m_denom!=0)
        return np.sum(f_m, axis=1)

@dataclass
class OdeintSolver(PythonSolver):

    @classmethod
    def func_sys(cls, r_0, m, r_v, t):
        r_v = r_v.reshape((-1, r_0.shape[1]))
        middle = r_v.shape[0] // 2
        result = np.zeros_like(r_v)
        result[:middle] = r_v[middle:]
        result[middle:] = cls.acceleration(r_v[:middle], m, r_v[:middle], m)
        return result.flatten()

    def solve(self, r_0, v_0, m_0):
        t_range = np.arange(0, self.dt * self.n_iters, self.dt)
        initial_conditions = np.concatenate((r_0, v_0)).flatten()
        func = functools.partial(self.func_sys, r_0, m_0)
        solution = odeint(func, initial_conditions, t_range)[:, :v_0.size]
        return solution.reshape((self.n_iters, -1, r_0.shape[1]))

@dataclass
class PythonVerletSolver(PythonSolver):

    @staticmethod
    def r_step(r, v, a, dt):
        return r + v * dt + 0.5 * a * dt * dt

    @staticmethod
    def v_step(v, a, next_a, dt):
        return v + 0.5 * (a + next_a) * dt

    def solve(self, r_0, v_0, m_0):
        result_shape = (self.n_iters, int(r_0.shape[0]) , 2)
        R = np.zeros(result_shape)
        V = R.copy()
        A = R.copy()

        R[0] = r_0
        V[0] = v_0
        A[0] = self.acceleration(r_0, m_0, r_0, m_0)

        for i in range(self.n_iters - 1):
            R[i + 1] = self.r_step(R[i], V[i], A[i], self.dt)
            A[i + 1] = self.acceleration(R[i + 1], m_0, R[i + 1], m_0)
            V[i + 1] = self.v_step(V[i], A[i], A[i + 1], self.dt)

        return R

@dataclass
class MultiprocessingVerletSolver(PythonVerletSolver):

    n_workers : int = 5

    def __post_init__(self):
        self.pool = mp.Pool(self.n_workers)

    def acceleration(self, r, m, _r, _m, G=6.6743 * 10 ** -11):
        splitted_r = np.split(r, self.n_workers)
        splitted_m = np.split(m, self.n_workers)

        func = functools.partial(super().acceleration, r_j=r, m_j=m)
        result = self.pool.starmap(func, zip(splitted_r, splitted_m))
        return np.vstack(result)
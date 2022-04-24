import numpy as np
cimport numpy as np
import cython

cdef class CythonSolver:

    def __init__(CythonSolver self, double dt, int n_iters, double G):
        self.dt = dt
        self.n_iters = n_iters
        self.G = G

    cpdef r_step(
        CythonSolver self,
        np.ndarray[np.float64_t, ndim=2] r, 
        np.ndarray[np.float64_t, ndim=2] v, 
        np.ndarray[np.float64_t, ndim=2] a, 
        double dt
        ):
        return r + v * dt + 0.5 * a * dt * dt

    cpdef v_step(
        CythonSolver self,
        np.ndarray[np.float64_t, ndim=2] v, 
        np.ndarray[np.float64_t, ndim=2] a, 
        np.ndarray[np.float64_t, ndim=2] next_a, 
        double dt
        ):
        return v + 0.5 * (a + next_a) * dt

    cpdef acceleration(
        CythonSolver self,
        np.ndarray[np.float64_t, ndim=2] r_i,
        np.ndarray[np.float64_t, ndim=1] m_i,
        np.ndarray[np.float64_t, ndim=2] r_j,
        np.ndarray[np.float64_t, ndim=1] m_j,
        double G
    ):
        cdef np.ndarray[np.float64_t, ndim=3] r_ij = r_j - r_i[:, None]

        cdef np.ndarray[np.float64_t, ndim=3] f_m_denom = np.linalg.norm(r_ij, axis=2)[...,None] ** 3
        cdef np.ndarray[np.float64_t, ndim=3] f_m = G * m_j[:, None] * np.divide(r_ij, f_m_denom, out=np.zeros_like(r_ij), where=f_m_denom!=0)

        return np.sum(f_m, axis=1)

    cpdef solve(
        CythonSolver self,
        np.ndarray[np.float64_t, ndim=2] r_0, 
        np.ndarray[np.float64_t, ndim=2] v_0, 
        np.ndarray[np.float64_t, ndim=1] m_0
        ):
        cdef tuple result_shape = (self.n_iters, int(r_0.shape[0]) , 2)
        cdef np.ndarray[np.float64_t, ndim=3] R = np.zeros(result_shape)
        cdef np.ndarray[np.float64_t, ndim=3] V = R.copy()
        cdef np.ndarray[np.float64_t, ndim=3] A = R.copy()

        R[0] = r_0
        V[0] = v_0
        A[0] = self.acceleration(r_0, m_0, r_0, m_0, self.G)

        for i in range(self.n_iters - 1):
            R[i + 1] = self.r_step(R[i], V[i], A[i], self.dt)
            A[i + 1] = self.acceleration(R[i + 1], m_0, R[i + 1], m_0, self.G)
            V[i + 1] = self.v_step(V[i], A[i], A[i + 1], self.dt)

        return R

#     dt : float
#     n_iters : int
#     G : float = 6.6743 * 10 ** -11

#     @staticmethod
#     def acceleration(r_i, m_i ,r_j, m_j, r_c, m_c, G=6.6743 * 10 ** -11):
#         r_ij = r_j - r_i[:, None]
#         r_ic = r_c - r_i[:, None]

#         f_m_denom = np.linalg.norm(r_ij, axis=2)[..., None] ** 3
#         f_m = G * m_j[:, None] * np.divide(r_ij, f_m_denom, out=np.zeros_like(r_ij), where=f_m_denom!=0)

#         f_c_denom = np.linalg.norm(r_ic, axis=2)[..., None] ** 3
#         f_c = G * m_c * np.divide(r_ic, f_c_denom, np.zeros_like(r_ic), where=f_c_denom!=0)

#         return (np.sum(f_m, axis=1) + np.sum(f_c, axis=1)).astype(np.float64)

# @dataclass
# class CythonVerletSolver(CythonSolver):

#     @staticmethod
#     def r_step(r, v, a, dt):
#         return r + v * dt + 0.5 * a * dt * dt

#     @staticmethod
#     def v_step(v, a, next_a, dt):
#         return v + 0.5 * (a + next_a) * dt

#     def solve(self, r_0, v_0, m_0, r_c, m_c):
#         result_shape = (self.n_iters, int(r_0.shape[0]) , 2)
#         R = np.zeros(result_shape)
#         V = R.copy()
#         A = R.copy()

#         R[0] = r_0
#         V[0] = v_0
#         A[0] = self.acceleration(r_0, m_0, r_0, m_0, r_c, m_c)

#         for i in range(self.n_iters - 1):
#             R[i + 1] = self.r_step(R[i], V[i], A[i], self.dt)
#             A[i + 1] = self.acceleration(R[i + 1], m_0, R[i + 1], m_0, r_c, m_c)
#             V[i + 1] = self.v_step(V[i], A[i], A[i + 1], self.dt)

#         return R
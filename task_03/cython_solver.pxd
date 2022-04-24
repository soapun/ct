
import numpy as np
cimport numpy as np

cdef class CythonSolver:

    cdef double dt
    cdef int n_iters
    cdef double G

    cpdef r_step(
        CythonSolver self,
        np.ndarray[np.float64_t, ndim=2] r, 
        np.ndarray[np.float64_t, ndim=2] v, 
        np.ndarray[np.float64_t, ndim=2] a, 
        double dt
    )

    cpdef v_step(
        CythonSolver self,
        np.ndarray[np.float64_t, ndim=2] v, 
        np.ndarray[np.float64_t, ndim=2] a, 
        np.ndarray[np.float64_t, ndim=2] next_a, 
        double dt
    )

    cpdef acceleration(
        CythonSolver self,
        np.ndarray[np.float64_t, ndim=2] r_i,
        np.ndarray[np.float64_t, ndim=1] m_i,
        np.ndarray[np.float64_t, ndim=2] r_j,
        np.ndarray[np.float64_t, ndim=1] m_j,
        double G
    )

    cpdef solve(
        CythonSolver self,
        np.ndarray[np.float64_t, ndim=2] r_0, 
        np.ndarray[np.float64_t, ndim=2] v_0, 
        np.ndarray[np.float64_t, ndim=1] m_0
    )
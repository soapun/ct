from dataclasses import dataclass
import numpy as np
import pyopencl as cl

@dataclass
class OpenCLSolver:
    dt : float
    n_iters : int
    G : float = 6.6743 * 10 ** -11
    group_size : int = 1

    def __post_init__(self):
        platform = cl.get_platforms()[0]
        device = platform.get_devices()[0]
        self.ctx = cl.Context([device])
        self.queue = cl.CommandQueue(self.ctx)
        

    def solve(self, r_0, v_0, m_0):
        self.program = cl.Program(self.ctx, f"#define N_DIM {int(r_0.shape[1])}\n" + open("opencl_solver.cl").read()).build()
        
        result_shape = (self.n_iters, int(r_0.shape[0]) , int(r_0.shape[1]))
        R = np.zeros(result_shape)
        V = R.copy()
        A = R.copy()
        R[0] = r_0
        V[0] = v_0

        R_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=R)
        V_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=V)
        A_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=A)
        m_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=m_0)
        localmem = cl.LocalMemory(self.group_size)

        self.program.solve(
            self.queue, 
            (r_0.shape[0],), 
            (self.group_size, ), 
            R_buf, V_buf, A_buf, m_buf, 
            np.double(self.G), np.double(self.dt), np.int32(r_0.shape[0]), 
            np.int32(r_0.shape[1]), np.int32(self.n_iters), localmem)

        R_np = np.empty_like(R)
        cl.enqueue_copy(self.queue, R_np, R_buf)
        return R_np
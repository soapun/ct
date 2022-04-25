__kernel void solve(__global double *R, __global double *V,
                              __global double *A, __global double *m,
                              const double G,
                              const double dt, const unsigned n_bodies,
                              const unsigned n_dim, const unsigned n_iters,
                              __local double *localmem) {
  int global_idx = get_global_id(0);

  for (int jb = 0; jb < n_bodies; jb++) {
    if (jb == global_idx) continue;
    double tmp[N_DIM];
    double norm = 0;
    for (size_t d=0; d < n_dim; d++) {
      tmp[d] = R[jb * n_dim + d] - R[global_idx * n_dim + d];
      norm += tmp[d] * tmp[d];
    }
    norm = rsqrt(norm);
    norm = norm * norm * norm;
    double f = G * norm * m[jb];
    for (size_t d=0; d < n_dim; d++) {
      A[global_idx * n_dim + d] += f * tmp[d];
    }
  }
  barrier(CLK_GLOBAL_MEM_FENCE);

  for (int i=1; i < n_iters; i++) {
    size_t cur = n_bodies * n_dim * i;
    size_t prev = n_bodies * n_dim * (i - 1);

    for (size_t d=0; d < n_dim; d++) {
      R[cur + global_idx * n_dim + d] = R[prev + global_idx * n_dim + d] + dt * V[prev + global_idx * n_dim + d] + 0.5 * dt * dt * A[prev + global_idx * n_dim + d];  
    }
    barrier(CLK_GLOBAL_MEM_FENCE);

    for (int jb = 0; jb < n_bodies; jb++) {
      if (jb == global_idx) continue;
      double tmp[N_DIM];
      double norm = 0;
      for (size_t d=0; d < n_dim; d++) {
        tmp[d] = R[cur + jb * n_dim + d] - R[cur + global_idx * n_dim + d];
        norm += tmp[d] * tmp[d];
      }
      norm = rsqrt(norm);
      norm = norm * norm * norm;
      double f = G * norm * m[jb];
      for (size_t d=0; d < n_dim; d++) {
        A[cur + global_idx * n_dim + d] += f * tmp[d];
      }
    }
    barrier(CLK_GLOBAL_MEM_FENCE);

    for (size_t d=0; d < n_dim; d++) {
      V[cur + global_idx * n_dim + d] = V[prev + global_idx * n_dim + d] + 0.5* dt * (A[prev + global_idx * n_dim + d] + A[cur + global_idx * n_dim + d]);
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
  }                          
}
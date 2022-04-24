__kernel void solve(__global double *R, __global double *V,
                              __global double *A, __global double *m,
                              const double G,
                              const double dt, const unsigned n_bodies,
                              const unsigned n_dim, const unsigned n_iters,
                              __local double *localmem) {
  int global_idx = get_global_id(0);
  int local_idx = get_local_id(0);
  int local_size = get_local_size(0);
  int n_work_groups = n_bodies / local_size;

  for (int jb = 0; jb < n_work_groups; jb++) {
      for (size_t d = 0; d < n_dim; d++) {
        localmem[local_idx * n_dim + d] = R[(jb * local_size + local_idx)* n_dim + d];
      }
      barrier(CLK_LOCAL_MEM_FENCE);

      double tmp[N_DIM];
      for(int j = 0; j < local_size; j++) {
          double norm = 0;
          for (size_t d=0; d < n_dim; d++) {
            tmp[d] = localmem[j * n_dim + d] - R[global_idx * n_dim + d];
            norm += tmp[d] * tmp[d];
          }
          norm = rsqrt(norm);
          norm = norm * norm * norm;
          double f = (global_idx != (jb * local_size + j)) ? G * norm * m[jb * local_size + j] : 0;

          for (size_t d=0; d < n_dim; d++) {
            A[global_idx * n_dim + d] += f * tmp[d];
          }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
  }

  for (int i=1; i < n_iters; i++) {
    size_t cur = n_bodies * n_dim * i;
    size_t prev = n_bodies * n_dim * (i - 1);

    for (size_t d=0; d < n_dim; d++) {
      R[cur + global_idx * n_dim + d] = R[prev + global_idx * n_dim + d] + dt * V[prev + global_idx * n_dim + d] + 0.5 * dt * dt * A[prev + global_idx * n_dim + d];  
    }

    for (int jb = 0; jb < n_work_groups; jb++) {
      for (size_t d = 0; d < n_dim; d++) {
        localmem[local_idx * n_dim + d] = R[cur + (jb * local_size + local_idx)* n_dim + d];
      }
      barrier(CLK_LOCAL_MEM_FENCE);

      double tmp[2];
      for(int j = 0; j < local_size; j++) {
          double norm = 0;
          for (size_t d=0; d < n_dim; d++) {
            tmp[d] = localmem[j * n_dim + d] - R[cur + global_idx * n_dim + d];
            norm += tmp[d] * tmp[d];
          }
          norm = rsqrt(norm);
          norm = norm * norm * norm;
          double f = (global_idx != (jb * local_size + j)) ? G * norm * m[jb * local_size + j] : 0;

          for (size_t d=0; d < n_dim; d++) {
            A[cur + global_idx * n_dim + d] += f * tmp[d];
          }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    for (size_t d=0; d < n_dim; d++) {
      V[cur + global_idx * n_dim + d] = V[prev + global_idx * n_dim + d] + 0.5* dt * (A[prev + global_idx * n_dim + d] + A[cur + global_idx * n_dim + d]);
    }

  }                              
}
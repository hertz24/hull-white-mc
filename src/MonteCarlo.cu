#include "../include/MonteCarlo.cuh"

__global__ void init_curand_state(unsigned long long seed, int n, curandState *state)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
        curand_init(seed, tid, 0, state + tid);
}

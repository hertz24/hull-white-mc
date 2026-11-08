#include "../include/HullWhite.cuh"

__global__ void init_curand_state(unsigned long long seed, int n, curandState *state)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
        curand_init(seed, tid, 0, state + tid);
}

__device__ float marketFitting(float t)
{
    float theta = 0.0f;
    if (t >= 0 && t < 5)
        theta = 0.012f + 0.0014f * t;
    else if (t >= 5 && t <= 10)
        theta = 0.019f + 0.001f * (t - 5);
    return theta;
}

__device__ float average(float r, float a, float t, float s)
{
    float dt = t - s;
    return r * expf(-a * dt) + (dt / 2) * (expf(-a * dt) * marketFitting(s) + marketFitting(t));
}

__device__ float uncertainty(float sigma, float a, float t, float s)
{
    return sqrt(sigma * sigma * (1 - exp(-2 * a * (t - s))) / (2 * a));
}

__global__ void shortRate(curandState *state, float *sum, float a, float sigma, float t, float s, int n, int m)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float tmp[1024];
    if (tid < m)
    {
        curandState localState = state[tid];
        float r = 0.012f;
        float dt = (t - s) / n;
        for (int i = 0; i < n; i++)
        {
            float G = curand_normal(&localState);
            t = s + dt;
            r = average(r, a, t, s) + uncertainty(sigma, a, t, s) * G;
            s = t;
        }
        state[tid] = localState;
        tmp[threadIdx.x] = r;
    }
    __syncthreads();
    for (int i = blockDim.x / 2; i != 0; i /= 2)
    {
        if (threadIdx.x < i)
            tmp[threadIdx.x] += tmp[threadIdx.x + i];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        atomicAdd(sum, tmp[0]);
}

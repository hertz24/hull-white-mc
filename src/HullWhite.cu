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
    return r * expf(-a * dt) + 0.5f * dt * (expf(-a * dt) * marketFitting(s) + marketFitting(t));
}

__device__ float uncertainty(float sigma, float a, float t, float s)
{
    return sqrt(sigma * sigma * (1 - exp(-2 * a * (t - s))) / (2 * a));
}

__device__ float shortRate(float r, float sigma, float a, float t, float s, float G)
{
    return average(r, a, t, s) + uncertainty(sigma, a, t, s) * G;
}

__global__ void zeroCouponBondPrice(curandState *state, float *sum, float sigma, float a, float t, float T, int N, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float tmp[1024];
    tmp[threadIdx.x] = 0.0f;
    if (tid < n)
    {
        curandState localState = state[tid];
        float r = 0.012f;
        float dt = (T - t) / N;
        float integral = 0.0f;
        float s = t;
        for (int i = 0; i < N; i++)
        {
            float G = curand_normal(&localState);
            float current_t = s + dt;
            float next_r = shortRate(r, sigma, a, current_t, s, G);
            integral += 0.5f * dt * (r + next_r);
            r = next_r;
            s = current_t;
        }
        state[tid] = localState;
        tmp[threadIdx.x] = expf(-integral);
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

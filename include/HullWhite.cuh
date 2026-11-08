#ifndef HULL_WHITE_H
#define HULL_WHITE_H

#include <curand_kernel.h>

__global__ void init_curand_state(unsigned long long seed, int n, curandState *state);

__device__ float marketFitting(float t);

__device__ float average(float r, float a, float t, float s);

__device__ float uncertainty(float sigma, float a, float t, float s);

__device__ float shortRate(float r, float sigma, float a, float t, float s, float G);

__global__ void zeroCouponBondPrice(curandState *state, float *sum, float sigma, float a, float t, float T, int N, int n);

#endif

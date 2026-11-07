#ifndef MONTE_CARLO_H
#define MONTE_CARLO_H

#include <curand_kernel.h>

__global__ void init_curand_state(unsigned long long seed, int n, curandState *state);

__global__ void monteCarloSimulation(curandState *state);

#endif

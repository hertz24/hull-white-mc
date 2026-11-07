#include "../include/HullWhite.cuh"

__device__ float uncertainty(float sigma, float a, float t, float s)
{
    return sqrt(sigma * sigma * (1 - exp(-2 * a * (t - s))) / (2 * a));
}

#include "CurandUtils.h"

#include <curand_kernel.h>

__global__ void initCurandState(curandState* curand_state, unsigned long long seed)
{
    curand_init(seed, 0, 0, curand_state);
}
#include "PDF.h"

__device__ PDF::PDF(curandState* curand_state)
    : curand_state(curand_state) {}
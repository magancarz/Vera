#pragma once

#include <cuda_runtime.h>

namespace dmm
{
    template <typename T, typename U, class... Us>
    __global__ void createObjectCUDA(T** object, Us... args)
    {
        (*object) = new U(args...);
    }

    template <typename T>
    __global__ void deleteObjectCUDA(T** object)
    {
        delete (*object);
    }
}
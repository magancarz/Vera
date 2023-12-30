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

    template <typename T>
    __global__ void deleteObjectCUDA(T** objects, size_t num_of_objects)
    {
        for(size_t i = 0; i < num_of_objects; ++i)
        {
            delete objects[i];
        }
    }
}
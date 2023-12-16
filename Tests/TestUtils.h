#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "RenderEngine/RayTracing/Shapes/Triangle.h"
#include "Utils/DeviceMemoryPointer.h"

//template <typename T>
//__global__ inline void deleteObjects(T** objects, int num)
//{
//    for (int i = 0; i < num; ++i)
//    {
//        delete objects[i];
//    }
//}

class TestUtils
{
public:
    static void createCurandState(curandState* curand_state);
    //static void createSetOfShapeMocks(dmm::DeviceMemoryPointer<Mesh>& mesh, dmm::DeviceMemoryPointer<Shape*>& shapes, dmm::DeviceMemoryPointer<ShapeInfo>& shape_infos);
};

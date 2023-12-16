#include "TestUtils.h"

#include "RenderEngine/RayTracing/Shapes/Triangle.h"

__global__ void initializeCurandState(curandState* curand_state)
{
    curand_init(2137, 0, 0, curand_state);
}

//__global__ void createSetOfShapeMocksCUDA(Mesh* parent, Triangle* shapes, ShapeInfo* shape_infos, int num_of_shapes)
//{
//    for (int i = 0; i < num_of_shapes; ++i)
//    {
//        shapes[i] = new Triangle(parent, glm::vec3{i % 4, i / 4, 0.f}, .25f);
//        shapes[i]->calculateObjectBounds();
//        shapes[i]->calculateWorldBounds();
//        shapes[i]->calculateShapeSurfaceArea();
//        shape_infos[i].object_bounds = shapes[i]->object_bounds;
//        shape_infos[i].world_bounds = shapes[i]->world_bounds;
//    }
//}

void TestUtils::createCurandState(curandState* curand_state)
{
    initializeCurandState<<<1, 1>>>(curand_state);
    cudaDeviceSynchronize();
}

//void TestUtils::createSetOfShapeMocks(dmm::DeviceMemoryPointer<Mesh>& mesh, dmm::DeviceMemoryPointer<Shape*>& shapes, dmm::DeviceMemoryPointer<ShapeInfo>& shape_infos)
//{
//    createSetOfShapeMocksCUDA<<<1, 1>>>(mesh.get(), shapes.get(), shape_infos.get(), shapes.getNumberOfElements());
//    cudaDeviceSynchronize();
//}
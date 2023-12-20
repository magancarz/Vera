#include "ConstantMedia.h"

#include "renderEngine/RayTracing/Shapes/Volume.h"
#include "Utils/CudaErrorChecker.h"
#include "Materials/MaterialAsset.h"
#include "helper_cuda.h"
#include "Materials/Material.h"
#include "models/AssetManager.h"
#include "Utils/CurandUtils.h"
#include "renderEngine/RayTracing/Shapes/ShapeInfo.h"

namespace cudaObjectUtils
{
    __global__ void createVolumesOnDeviceMemory(
        Object* parent,
        glm::mat4* object_to_world, glm::mat4* world_to_object,
        Bounds3f bounds, float density, Shape** cuda_shapes,
        ShapeInfo* shapes_infos,
        Material* material,
        size_t* next_triangle_id)
    {
        cuda_shapes[0] = new Volume(parent, (*next_triangle_id)++, material, bounds, density);
        transformShape(object_to_world, world_to_object, cuda_shapes[0], &shapes_infos[0]);
    }
}

ConstantMedia::ConstantMedia(Scene* parent_scene, std::shared_ptr<RawModel> model_data, Bounds3f bounds, float density)
    : Object(parent_scene, AssetManager::findMaterialAsset("barrel"), std::move(model_data), glm::vec3{bounds.min + bounds.max} / 2.f, glm::vec3{0}, 1.f), bounds(bounds), density(density) {}

void ConstantMedia::createShapesOnDeviceMemory()
{
    num_of_shapes = 1;
    shapes = dmm::DeviceMemoryPointer<Shape*>(num_of_shapes);
    shapes_infos = dmm::DeviceMemoryPointer<ShapeInfo>(num_of_shapes);

    cudaObjectUtils::createVolumesOnDeviceMemory<<<1, 1>>>(
        this,
        object_to_world.data(), world_to_object.data(),
        bounds, density, shapes.data(), shapes_infos.data(),
        nullptr, next_shape_id.data());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

bool ConstantMedia::determineIfShapeIsEmittingLight(size_t i)
{
    return false;
}

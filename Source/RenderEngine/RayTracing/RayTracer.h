#pragma once

#include "RayTracedImage.h"
#include "RayTracerCamera.h"
#include "Utils/DeviceMemoryPointer.h"

class Scene;
class AssetManager;
class Object;
class Camera;

class RayTracer
{
public:
    static void prepareCudaDevice();

    std::weak_ptr<Object> traceRayFromMouse(Scene* scene, const std::shared_ptr<Camera>& editor_camera, int x, int y);
    void generateRayTracedImage(Scene* scene, const std::shared_ptr<Camera>& editor_camera, const std::shared_ptr<RayTracedImage>& current_image);
    void runRayTracer(Scene* scene, const std::shared_ptr<RayTracedImage>& current_image, const dim3& blocks, const dim3& threads_per_block);

protected:
    dmm::DeviceMemoryPointer<RayTracerCamera> cuda_camera;
    const unsigned int block_size = 4;
    const dim3 threads_per_block{block_size, block_size};
};

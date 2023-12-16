#include "RayTracer.h"

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <cuda/helper_cuda.h>

#include <chrono>

#include "RayTracerCamera.h"
#include "Objects/Object.h"
#include "Scene/Scene.h"
#include "Objects/Camera.h"
#include "Utils/DeviceMemoryPointer.h"
#include "IntersectionAccelerators/BVHTreeTraverser.h"

#include "GUI/Display.h"

namespace RayTracing
{
    __global__ void findObjectIntersectingRay(BVHTreeTraverser* intersection_accelerator_tree_traverser, RayTracerCamera* camera, int x, int y, int image_width, int image_height, Object** found_mesh, bool* found)
    {
        const auto image_width_reciprocal = 1.f / static_cast<float>(image_width);
        const auto image_height_reciprocal = 1.f / static_cast<float>(image_height);
        const float u = x * image_width_reciprocal;
        const float v = (image_height - y) * image_height_reciprocal;
        Ray ray = camera->getRay(u, v);
        const HitRecord rec = intersection_accelerator_tree_traverser->checkIntersection(&ray);
        (*found_mesh) = rec.parent_object;
        (*found) = found_mesh != nullptr;
    }
}

void RayTracer::prepareCudaDevice()
{
    checkCudaErrors(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1073741824));
    checkCudaErrors(cudaDeviceSetLimit(cudaLimitStackSize, 4056));
}

std::weak_ptr<Object> RayTracer::traceRayFromMouse(Scene* scene, const std::shared_ptr<Camera>& editor_camera, int x, int y)
{
    constexpr float aspect = static_cast<float>(Display::WINDOW_WIDTH) / static_cast<float>(Display::WINDOW_HEIGHT);
    const RayTracerCamera ray_tracing_editor_camera
    {
        editor_camera->getPosition(),
        editor_camera->getDirection(),
        editor_camera->getWorldUpVector(),
        editor_camera->getFieldOfView(),
        aspect,
    };

    dmm::DeviceMemoryPointer<Object*> object;
    dmm::DeviceMemoryPointer<bool> found;
    cuda_camera.copyFrom(&ray_tracing_editor_camera);
    RayTracing::findObjectIntersectingRay<<<1, 1>>>(scene->intersection_accelerator_tree_traverser.data(), cuda_camera.data(), x, y, Display::WINDOW_WIDTH, Display::WINDOW_HEIGHT, object.data(), found.data());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    std::weak_ptr<Object> result{};
    if (*found)
    {
       result = scene->findObjectByID((*object)->object_id);
    }
    return result;
}

void RayTracer::generateRayTracedImage(Scene* scene, const std::shared_ptr<Camera>& editor_camera, const std::shared_ptr<RayTracedImage>& current_image)
{
    const float aspect = static_cast<float>(current_image->image_config.image_width) / static_cast<float>(current_image->image_config.image_height);
    const RayTracerCamera ray_tracing_editor_camera
    {
        editor_camera->getPosition(),
        editor_camera->getDirection(),
        editor_camera->getWorldUpVector(),
        editor_camera->getFieldOfView(),
        aspect,
        current_image->image_config.aperture,
        current_image->image_config.focus_dist
    };
    cuda_camera.copyFrom(&ray_tracing_editor_camera);

    ++current_image->generated_samples;

    const dim3 blocks(current_image->image_config.image_width / block_size + 1, current_image->image_config.image_height / block_size + 1);
    const auto start = std::chrono::steady_clock::now();
    runRayTracer(scene, current_image, blocks, threads_per_block);
    const auto end = std::chrono::steady_clock::now();
    std::cout << "Rendering time: " <<
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() <<
        " milliseconds." << std::endl;

    current_image->updateImage();
}
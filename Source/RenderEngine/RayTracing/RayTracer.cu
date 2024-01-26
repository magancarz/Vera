#include "RayTracer.h"

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <cuda/helper_cuda.h>

#include <chrono>

#include "RayTracerCamera.h"
#include "Objects/Object.h"
#include "Scene/Scene.h"
#include "renderEngine/Camera.h"
#include "Utils/DeviceMemoryPointer.h"
#include "IntersectionAccelerators/BVHTreeTraverser.h"
#include "GUI/Display.h"
#include "renderEngine/RayTracing/PDF/HittablePDF.h"
#include "renderEngine/RayTracing/PDF/MixturePDF.h"
#include "ScatterRecord.h"

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
        (*found_mesh) = (*rec.intersected_shape)->parent;
        (*found) = (*found_mesh) != nullptr;
    }

    __device__ glm::vec3 gatherColorInformationFromSceneIntersectionWithHybridRayTracing(BVHTreeTraverser* intersection_accelerator_tree_traverser, Shape** sampled_shapes, int num_of_sampled_shapes, Ray* ray, int depth)
    {
        HittablePDF hittable_pdf(ray->curand_state, intersection_accelerator_tree_traverser, sampled_shapes, num_of_sampled_shapes);
        glm::vec3 color{1.f};
        int max_iterations = 50;
        max_iterations = depth > max_iterations ? depth * 2 : max_iterations;
        for (int current_depth = depth, current_iteration = 0; current_depth >= 0 && current_iteration < max_iterations; --current_depth, ++current_iteration)
        {
            if (depth == 0)
            {
                return glm::vec3{0};
            }

            HitRecord rec = intersection_accelerator_tree_traverser->checkIntersection(ray);
            if (rec.did_hit_anything)
            {
                ScatterRecord scatter_record{};
                if ((*rec.intersected_shape)->scatter(ray, &rec, &scatter_record))
                {
                    if (scatter_record.is_specular)
                    {
                        scatter_record.specular_ray.curand_state = ray->curand_state;
                        *ray = scatter_record.specular_ray;
                        color *= scatter_record.color;
                        ++current_depth;
                        continue;
                    }

                    hittable_pdf.changeHitRecord(&rec);
                    MixturePDF mixture_pdf{
                            ray->curand_state,
                            hittable_pdf,
                            scatter_record.pdf};
                    Ray scattered{rec.hit_point, mixture_pdf.generate()};
                    auto pdf = mixture_pdf.value(scattered.direction);
                    auto scattering_pdf = (*rec.intersected_shape)->scatteringPDF(&rec, &scattered);

                    scattered.curand_state = ray->curand_state;
                    *ray = scattered;
                    color *= scatter_record.color * scattering_pdf / pdf;
                    continue;
                }

                return color * (*rec.intersected_shape)->emitted(rec.uv);
            }

            float t = 0.5f * (ray->direction.y + 1.f);
            return (1.f - t) * glm::vec3{1} + t * glm::vec3{0.5f, 0.7f, 1.0f};
        }

        return glm::vec3{0.f};
    }

    __global__ void generateImageWithHybridRayTracing(unsigned char* cuda_texture_array, unsigned long* accumulated_texture_data, int number_of_total_samples, RayTracerCamera* camera, BVHTreeTraverser* intersection_accelerator_tree_traverser, Shape** sampled_shapes, int num_of_sampled_shapes, int depth, int image_width, int image_height)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x < image_width && y < image_height)
        {
            constexpr int samples_per_pixel = 5;
            const auto image_width_reciprocal = 1.f / static_cast<float>(image_width);
            const auto image_height_reciprocal = 1.f / static_cast<float>(image_height);

            auto curand_state = new curandState();
            curand_init(cuda::std::chrono::high_resolution_clock::now().time_since_epoch().count(), x + y * blockDim.x, 0, curand_state);

            glm::vec3 color{0};
            for (int i = 0; i < samples_per_pixel; ++i)
            {
                const float u = (static_cast<float>(x) + curand_uniform(curand_state) - 0.5f) * image_width_reciprocal;
                const float v = (static_cast<float>(image_height - y) + curand_uniform(curand_state) - 0.5f) * image_height_reciprocal;
                Ray ray = camera->getRay(curand_state, u, v);
                ray.curand_state = curand_state;
                color += gatherColorInformationFromSceneIntersectionWithHybridRayTracing(intersection_accelerator_tree_traverser, sampled_shapes, num_of_sampled_shapes, &ray, depth);
            }

            delete curand_state;

            color /= static_cast<float>(samples_per_pixel);
            color = glm::vec3{1.0f} - glm::exp(-color * 2.f);

            const size_t index = y * image_width + x;

            const auto ir = static_cast<unsigned char>(255.99f * color.x);
            const auto ig = static_cast<unsigned char>(255.99f * color.y);
            const auto ib = static_cast<unsigned char>(255.99f * color.z);

            accumulated_texture_data[index * 3 + 0] += ir;
            accumulated_texture_data[index * 3 + 1] += ig;
            accumulated_texture_data[index * 3 + 2] += ib;

            cuda_texture_array[index * 3 + 0] = static_cast<float>(accumulated_texture_data[index * 3 + 0]) / static_cast<float>(number_of_total_samples);
            cuda_texture_array[index * 3 + 1] = static_cast<float>(accumulated_texture_data[index * 3 + 1]) / static_cast<float>(number_of_total_samples);
            cuda_texture_array[index * 3 + 2] = static_cast<float>(accumulated_texture_data[index * 3 + 2]) / static_cast<float>(number_of_total_samples);
        }
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
        editor_camera->getCameraFrontVector(),
        editor_camera->getCameraUpVector(),
        editor_camera->getFieldOfView(),
        aspect,
    };

    dmm::DeviceMemoryPointer<Object*> object;
    dmm::DeviceMemoryPointer<bool> found;
    found.copyFrom(false);
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
        editor_camera->getCameraFrontVector(),
        editor_camera->getCameraUpVector(),
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
        " milliseconds.\n";

    current_image->updateImage();
}

void RayTracer::runRayTracer(Scene* scene, const std::shared_ptr<RayTracedImage>& current_image, const dim3& blocks, const dim3& threads_per_block)
{
    RayTracing::generateImageWithHybridRayTracing<<<blocks, threads_per_block>>>(
current_image->texture_data_ptr,
        current_image->accumulated_texture_data.data(),
        current_image->generated_samples,
        cuda_camera.data(),
        scene->intersection_accelerator_tree_traverser.data(),
        scene->scene_light_sources.data(),
        scene->scene_light_sources.size(),
        current_image->image_config.number_of_ray_bounces,
        current_image->image_config.image_width,
        current_image->image_config.image_height);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}
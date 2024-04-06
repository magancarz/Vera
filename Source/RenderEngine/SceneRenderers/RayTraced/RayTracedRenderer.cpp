#include "RayTracedRenderer.h"

RayTracedRenderer::RayTracedRenderer(Device& device, World* world)
    : device{device}, world{world}
{
    queryRayTracingPipelineProperties();
    createBottomLevelAccelerationStructure();
}

void RayTracedRenderer::queryRayTracingPipelineProperties()
{
    VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    properties.pNext = &ray_tracing_properties;
    vkGetPhysicalDeviceProperties2(device.getPhysicalDevice(), &properties);
}

void RayTracedRenderer::createBottomLevelAccelerationStructure()
{
    ray_tracing_builder.setup(device.findPhysicalQueueFamilies().graphicsFamily, ray_tracing_properties);

    std::vector<RayTracingBuilder::BlasInput> all_blas_inputs;
    all_blas_inputs.reserve(world->objects.size());
    for (auto& [id, object] : world->objects)
    {
        RayTracingBuilder::BlasInput blas_input = object->model->getBlasInput();
        all_blas_inputs.emplace_back(blas_input);
    }

    ray_tracing_builder.buildBlas(device, all_blas_inputs[0]);
    ray_tracing_builder.buildTlas(device);
}
#include "RayTracedRenderer.h"

#include <chrono>
#include <iostream>

#include "RenderEngine/RenderingAPI/VulkanHelper.h"
#include "RenderEngine/GlobalUBO.h"
#include "RenderEngine/RenderingAPI/VulkanDefines.h"

RayTracedRenderer::RayTracedRenderer(Device& device, World* world)
    : device{device}, world{world}
{
    queryRayTracingPipelineProperties();
    createAccelerationStructure();
    createRayTracedImage();
    createCameraUniformBuffer();
    createObjectDescriptionsBuffer();
    createLightIndicesBuffer();
    createDescriptors();
    createRayTracingPipeline();
}

void RayTracedRenderer::queryRayTracingPipelineProperties()
{
    VkPhysicalDevice physical_device = device.getPhysicalDevice();

    VkPhysicalDeviceProperties physical_device_properties;
    vkGetPhysicalDeviceProperties(physical_device,
                                  &physical_device_properties);

    ray_tracing_properties = VkPhysicalDeviceRayTracingPipelinePropertiesKHR{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};
    VkPhysicalDeviceProperties2 physical_device_properties_2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    physical_device_properties_2.pNext = &ray_tracing_properties;
    physical_device_properties_2.properties = physical_device_properties;
    vkGetPhysicalDeviceProperties2(physical_device, &physical_device_properties_2);
}

void RayTracedRenderer::createAccelerationStructure()
{
    RayTracingAccelerationStructureBuilder builder{device};

    std::vector<BlasInstance*> blas_instances;
    blas_instances.reserve(world->objects.size());
    size_t i = 0;
    for (auto [_, object] : world->objects)
    {
        auto blas_instance = object->getBlasInstance();
        blas_instance->bottomLevelAccelerationStructureInstance.instanceCustomIndex = i++;
        blas_instances.push_back(blas_instance);
    }
    tlas = builder.buildTopLevelAccelerationStructure(blas_instances);
}

void RayTracedRenderer::createRayTracedImage()
{
    VkSurfaceCapabilitiesKHR surface_capabilities;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device.getPhysicalDevice(), device.surface(), &surface_capabilities);
    ray_traced_texture = std::make_unique<Texture>(
            device,
            surface_capabilities.currentExtent.width,
            surface_capabilities.currentExtent.height,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_FORMAT_R8G8B8A8_UNORM);
}

void RayTracedRenderer::createCameraUniformBuffer()
{
    camera_uniform_buffer = std::make_unique<Buffer>
    (
            device,
            sizeof(GlobalUBO),
            1,
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
    );
    camera_uniform_buffer->map();
}

void RayTracedRenderer::createObjectDescriptionsBuffer()
{
    object_descriptions_buffer = std::make_unique<Buffer>
    (
            device,
            sizeof(ObjectDescription),
            static_cast<uint32_t>(world->objects.size()),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT
    );
    std::vector<ObjectDescription> object_descriptions;
    object_descriptions.reserve(world->objects.size());
    for (auto& [_, object] : world->objects)
    {
        object_descriptions.emplace_back(object->getObjectDescription());
    }
    object_descriptions_buffer->writeWithStagingBuffer(object_descriptions.data());
}

void RayTracedRenderer::createLightIndicesBuffer()
{
    std::vector<uint32_t> light_indices;
    size_t i = 0;
    for (auto& [_, object] : world->objects)
    {
        if (object->isLightObject())
        {
            light_indices.push_back(i);
        }
        ++i;
    }
    number_of_lights = light_indices.size();
    light_indices_buffer = std::make_unique<Buffer>
    (
            device,
            sizeof(uint32_t),
            static_cast<uint32_t>(light_indices.size()),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT
    );
    light_indices_buffer->writeWithStagingBuffer(light_indices.data());
}

void RayTracedRenderer::createDescriptors()
{
    descriptor_pool = DescriptorPool::Builder(device)
            .setMaxSets(2)
            .addPoolSize(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1)
            .addPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1)
            .addPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3)
            .addPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1)
            .build();

    descriptor_set_layout = DescriptorSetLayout::Builder(device)
            .addBinding(0, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR)
            .addBinding(1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_RAYGEN_BIT_KHR)
            .addBinding(2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR)
            .addBinding(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR)
            .addBinding(4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR)
            .build();

    VkWriteDescriptorSetAccelerationStructureKHR
            acceleration_structure_descriptor_info = {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
            .pNext = NULL,
            .accelerationStructureCount = 1,
            .pAccelerationStructures = &tlas.acceleration_structure};

    auto camera_buffer_descriptor_info = camera_uniform_buffer->descriptorInfo();
    auto object_descriptions_buffer_descriptor_info = object_descriptions_buffer->descriptorInfo();
    auto light_indices_descriptor_info = light_indices_buffer->descriptorInfo();

    VkDescriptorImageInfo rayTraceImageDescriptorInfo = {
            .sampler = VK_NULL_HANDLE,
            .imageView = ray_traced_texture->getImageView(),
            .imageLayout = VK_IMAGE_LAYOUT_GENERAL};

    DescriptorWriter(*descriptor_set_layout, *descriptor_pool)
            .writeAccelerationStructure(0, &acceleration_structure_descriptor_info)
            .writeImage(1, &rayTraceImageDescriptorInfo)
            .writeBuffer(2, &camera_buffer_descriptor_info)
            .writeBuffer(3, &object_descriptions_buffer_descriptor_info)
            .writeBuffer(4, &light_indices_descriptor_info)
            .build(descriptor_set_handle);
}

void RayTracedRenderer::createRayTracingPipeline()
{
    ray_tracing_pipeline = std::make_unique<RayTracingPipeline>(device, descriptor_set_layout->getDescriptorSetLayout(), ray_tracing_properties);
}

void RayTracedRenderer::renderScene(FrameInfo& frame_info)
{
    ray_tracing_pipeline->bind(frame_info.command_buffer);
    updatePipelineUniformVariables(frame_info);
    executeRayTracing(frame_info);
}

void RayTracedRenderer::updatePipelineUniformVariables(FrameInfo& frame_info)
{
    updateCameraUniformBuffer(frame_info);
    updateRayPushConstant(frame_info);
}

void RayTracedRenderer::updateCameraUniformBuffer(FrameInfo& frame_info)
{
    CameraUBO camera_ubo{};
    camera_ubo.camera_view = frame_info.camera->getView();
    camera_ubo.camera_proj = frame_info.camera->getProjection();
    camera_uniform_buffer->writeToBuffer(&camera_ubo);
    camera_uniform_buffer->flush();
    ray_tracing_pipeline->bindDescriptorSets(frame_info.command_buffer, {descriptor_set_handle});
}

void RayTracedRenderer::updateRayPushConstant(FrameInfo& frame_info)
{
    PushConstantRay push_constant_ray{};
    push_constant_ray.time = std::chrono::system_clock::now().time_since_epoch().count();
    current_number_of_frames = frame_info.need_to_refresh_generated_image ? 0 : current_number_of_frames;
    push_constant_ray.frames = current_number_of_frames;
    push_constant_ray.number_of_lights = number_of_lights;
    push_constant_ray.weather = frame_info.weather;
    push_constant_ray.sun_position = frame_info.sun_position;
    ray_tracing_pipeline->pushConstants(frame_info.command_buffer, push_constant_ray);
}

void RayTracedRenderer::executeRayTracing(FrameInfo& frame_info)
{
    constexpr uint32_t DEPTH = 1;
    pvkCmdTraceRaysKHR(frame_info.command_buffer,
            &ray_tracing_pipeline->rgenShaderBindingTable,
            &ray_tracing_pipeline->rmissShaderBindingTable,
            &ray_tracing_pipeline->rchitShaderBindingTable,
            &ray_tracing_pipeline->callableShaderBindingTable,
            frame_info.window_size.width, frame_info.window_size.height, DEPTH);

    ++current_number_of_frames;
}
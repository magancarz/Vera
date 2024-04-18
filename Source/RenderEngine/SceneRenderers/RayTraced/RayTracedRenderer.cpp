#include "RayTracedRenderer.h"

#include <chrono>
#include <iostream>

#include "RenderEngine/RenderingAPI/VulkanHelper.h"
#include "RenderEngine/GlobalUBO.h"

RayTracedRenderer::RayTracedRenderer(Device& device, World* world)
    : device{device}, world{world}
{
    queryRayTracingPipelineProperties();
    createAccelerationStructure();
    createRayTracedImage();
    createCameraUniformBuffer();
    createMaterialsBuffer();
    createObjectDescriptionsBuffer();
    createDescriptors();
    createRayTracingPipeline();
}

void RayTracedRenderer::queryRayTracingPipelineProperties()
{
    VkPhysicalDevice activePhysicalDeviceHandle = device.getPhysicalDevice();

    VkPhysicalDeviceProperties physicalDeviceProperties;
    vkGetPhysicalDeviceProperties(activePhysicalDeviceHandle,
                                  &physicalDeviceProperties);

    ray_tracing_properties = VkPhysicalDeviceRayTracingPipelinePropertiesKHR{
            .sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR,
            .pNext = NULL};

    VkPhysicalDeviceProperties2 physicalDeviceProperties2 = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
            .pNext = &ray_tracing_properties,
            .properties = physicalDeviceProperties};

    vkGetPhysicalDeviceProperties2(activePhysicalDeviceHandle,
                                   &physicalDeviceProperties2);
}

void RayTracedRenderer::createAccelerationStructure()
{
    RayTracingAccelerationStructureBuilder builder{device};

    std::vector<BlasInstance*> blas_instances;
    blas_instances.reserve(world->objects.size());
    std::transform(world->objects.begin(), world->objects.end(), std::back_inserter(blas_instances),
                   [](const std::pair<uint32_t, std::shared_ptr<Object>>& obj_pair) { return obj_pair.second->getBlasInstance(); });
    tlas = builder.buildTopLevelAccelerationStructure(blas_instances);
}

void RayTracedRenderer::createRayTracedImage()
{
    VkResult result;
    auto graphics_queue_index = device.findPhysicalQueueFamilies().graphicsFamily;

    uint32_t surfaceFormatCount = 0;
    result = vkGetPhysicalDeviceSurfaceFormatsKHR(
            device.getPhysicalDevice(), device.surface(), &surfaceFormatCount, NULL);

    std::vector<VkSurfaceFormatKHR> surfaceFormatList(surfaceFormatCount);
    result = vkGetPhysicalDeviceSurfaceFormatsKHR(
            device.getPhysicalDevice(), device.surface(), &surfaceFormatCount,
            surfaceFormatList.data());

    VkSurfaceCapabilitiesKHR surfaceCapabilities;
    result = vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
            device.getPhysicalDevice(), device.surface(), &surfaceCapabilities);

    VkImageCreateInfo rayTraceImageCreateInfo = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .pNext = NULL,
            .flags = 0,
            .imageType = VK_IMAGE_TYPE_2D,
            .format = surfaceFormatList[0].format,
            .extent = {.width = surfaceCapabilities.currentExtent.width,
                    .height = surfaceCapabilities.currentExtent.height,
                    .depth = 1},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 1,
            .pQueueFamilyIndices = &graphics_queue_index,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};

    result = vkCreateImage(device.getDevice(), &rayTraceImageCreateInfo, NULL,
                           &rayTraceImageHandle);

    if (result != VK_SUCCESS) {
        throw std::runtime_error("vkCreateImage");
    }

    VkMemoryRequirements rayTraceImageMemoryRequirements;
    vkGetImageMemoryRequirements(device.getDevice(), rayTraceImageHandle,
                                 &rayTraceImageMemoryRequirements);

    uint32_t rayTraceImageMemoryTypeIndex = device.findMemoryType(rayTraceImageMemoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    VkMemoryAllocateInfo rayTraceImageMemoryAllocateInfo = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .pNext = NULL,
            .allocationSize = rayTraceImageMemoryRequirements.size,
            .memoryTypeIndex = rayTraceImageMemoryTypeIndex};

    VkDeviceMemory rayTraceImageDeviceMemoryHandle = VK_NULL_HANDLE;
    result = vkAllocateMemory(device.getDevice(), &rayTraceImageMemoryAllocateInfo,
                              NULL, &rayTraceImageDeviceMemoryHandle);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("vkAllocateMemory");
    }

    result = vkBindImageMemory(device.getDevice(), rayTraceImageHandle,
                               rayTraceImageDeviceMemoryHandle, 0);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("vkBindImageMemory");
    }

    VkImageViewCreateInfo rayTraceImageViewCreateInfo = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .pNext = NULL,
            .flags = 0,
            .image = rayTraceImageHandle,
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = surfaceFormatList[0].format,
            .components = {.r = VK_COMPONENT_SWIZZLE_IDENTITY,
                    .g = VK_COMPONENT_SWIZZLE_IDENTITY,
                    .b = VK_COMPONENT_SWIZZLE_IDENTITY,
                    .a = VK_COMPONENT_SWIZZLE_IDENTITY},
            .subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = 1}};

    result = vkCreateImageView(device.getDevice(), &rayTraceImageViewCreateInfo, NULL,
                               &rayTraceImageViewHandle);

    if (result != VK_SUCCESS) {
        throw std::runtime_error("vkCreateImageView");
    }

    // =========================================================================
    // Ray Trace Image Barrier
    // (VK_IMAGE_LAYOUT_UNDEFINED -> VK_IMAGE_LAYOUT_GENERAL)

    auto command_buffer = device.beginSingleTimeCommands();

    VkImageMemoryBarrier rayTraceGeneralMemoryBarrier = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .pNext = NULL,
            .srcAccessMask = 0,
            .dstAccessMask = 0,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_GENERAL,
            .srcQueueFamilyIndex = graphics_queue_index,
            .dstQueueFamilyIndex = graphics_queue_index,
            .image = rayTraceImageHandle,
            .subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = 1}};

    vkCmdPipelineBarrier(command_buffer,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, NULL, 0, NULL,
                         1, &rayTraceGeneralMemoryBarrier);

    device.endSingleTimeCommands(command_buffer);
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

void RayTracedRenderer::createMaterialsBuffer()
{
    materials.push_back(Material{.color = glm::vec3{0.8, 0.8, 0.8}});
    materials.push_back(Material{.color = glm::vec3{0.8, 0.8, 0.8}});
    materials.push_back(Material{.color = glm::vec3{0.8, 0.8, 0.8}});
    materials.push_back(Material{.color = glm::vec3{0.8, 0.8, 0.8}});
    materials.push_back(Material{.color = glm::vec3{0.8, 0.8, 0.8}});
    materials.push_back(Material{.color = glm::vec3{0.0, 1.0, 0.0}});
    materials.push_back(Material{.color = glm::vec3{1.0, 1.0, 1.0}, .brightness = 1});
    materials.push_back(Material{.color = glm::vec3{1.0, 0.0, 0.0}});

    material_uniform_buffer = std::make_unique<Buffer>
    (
            device,
            sizeof(Material),
            static_cast<uint32_t>(materials.size()),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT
    );

    material_uniform_buffer->writeWithStagingBuffer(materials.data());
}

void RayTracedRenderer::createObjectDescriptionsBuffer()
{
    object_descriptions_buffer = std::make_unique<Buffer>
    (
            device,
            sizeof(ObjectDescription),
            static_cast<uint32_t>(world->objects.size() + 1),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT
    );
    std::vector<ObjectDescription> object_descriptions;
    object_descriptions.resize(world->objects.size() + 1);
    for (auto& [id, object] : world->objects)
    {
        object_descriptions[id] = object->getObjectDescription();
    }
    object_descriptions_buffer->writeWithStagingBuffer(object_descriptions.data());
}

void RayTracedRenderer::createDescriptors()
{
    descriptor_pool = DescriptorPool::Builder(device)
            .setMaxSets(2)
            .addPoolSize(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1)
            .addPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1)
            .addPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4)
            .addPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1)
            .build();

    descriptor_set_layout = DescriptorSetLayout::Builder(device)
            .addBinding(0, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR)
            .addBinding(1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_RAYGEN_BIT_KHR)
            .addBinding(2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR)
            .addBinding(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR)
            .build();

    material_descriptor_set_layout = DescriptorSetLayout::Builder(device)
            .addBinding(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR)
            .build();

    VkWriteDescriptorSetAccelerationStructureKHR
            accelerationStructureDescriptorInfo = {
            .sType =
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
            .pNext = NULL,
            .accelerationStructureCount = 1,
            .pAccelerationStructures = &tlas.acceleration_structure};

    auto camera_buffer_descriptor_info = camera_uniform_buffer->descriptorInfo();
    auto object_descriptions_buffer_descriptor_info = object_descriptions_buffer->descriptorInfo();

    VkDescriptorImageInfo rayTraceImageDescriptorInfo = {
            .sampler = VK_NULL_HANDLE,
            .imageView = rayTraceImageViewHandle,
            .imageLayout = VK_IMAGE_LAYOUT_GENERAL};

    DescriptorWriter(*descriptor_set_layout, *descriptor_pool)
            .writeAccelerationStructure(0, &accelerationStructureDescriptorInfo)
            .writeImage(1, &rayTraceImageDescriptorInfo)
            .writeBuffer(2, &camera_buffer_descriptor_info)
            .writeBuffer(3, &object_descriptions_buffer_descriptor_info)
            .build(descriptor_set_handle);

    auto material_uniform_buffer_descriptor_info = material_uniform_buffer->descriptorInfo();

    DescriptorWriter(*material_descriptor_set_layout, *descriptor_pool)
            .writeBuffer(0, &material_uniform_buffer_descriptor_info)
            .build(material_descriptor_set_handle);
}

void RayTracedRenderer::createRayTracingPipeline()
{
    ray_tracing_pipeline = std::make_unique<RayTracingPipeline>(device, descriptor_set_layout->getDescriptorSetLayout(), material_descriptor_set_layout->getDescriptorSetLayout(), ray_tracing_properties);
}

void RayTracedRenderer::renderScene(FrameInfo& frame_info)
{
    CameraUBO camera_ubo{};
    camera_ubo.camera_view = frame_info.camera->getView();
    camera_ubo.camera_proj = frame_info.camera->getProjection();
    camera_uniform_buffer->writeToBuffer(&camera_ubo);
    camera_uniform_buffer->flush();

    ray_tracing_pipeline->bind(frame_info.command_buffer);
    ray_tracing_pipeline->bindDescriptorSets(frame_info.command_buffer, {descriptor_set_handle, material_descriptor_set_handle});

    PushConstantRay push_constant_ray{};
    push_constant_ray.time = std::chrono::system_clock::now().time_since_epoch().count();
    current_number_of_frames = frame_info.player_moved ? 0 : current_number_of_frames;
    push_constant_ray.frames = current_number_of_frames;
    ray_tracing_pipeline->pushConstants(frame_info.command_buffer, push_constant_ray);

    pvkCmdTraceRaysKHR(frame_info.command_buffer,
            &ray_tracing_pipeline->rgenShaderBindingTable,
            &ray_tracing_pipeline->rmissShaderBindingTable,
            &ray_tracing_pipeline->rchitShaderBindingTable,
            &ray_tracing_pipeline->callableShaderBindingTable,
            //TODO: not hardcoded
            1280, 800, 1);

    auto graphics_queue_index = device.findPhysicalQueueFamilies().graphicsFamily;

    VkImageMemoryBarrier swapchainCopyMemoryBarrier = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .pNext = NULL,
            .srcAccessMask = 0,
            .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .srcQueueFamilyIndex = graphics_queue_index,
            .dstQueueFamilyIndex = graphics_queue_index,
            .image = frame_info.swap_chain_image,
            .subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = 1}};

    vkCmdPipelineBarrier(frame_info.command_buffer,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, NULL, 0,
                         NULL, 1, &swapchainCopyMemoryBarrier);

    VkImageMemoryBarrier rayTraceCopyMemoryBarrier = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .pNext = NULL,
            .srcAccessMask = 0,
            .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
            .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            .srcQueueFamilyIndex = graphics_queue_index,
            .dstQueueFamilyIndex = graphics_queue_index,
            .image = rayTraceImageHandle,
            .subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = 1}};

    vkCmdPipelineBarrier(frame_info.command_buffer,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, NULL, 0,
                         NULL, 1, &rayTraceCopyMemoryBarrier);

    VkImageCopy imageCopy = {
            .srcSubresource = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                    .mipLevel = 0,
                    .baseArrayLayer = 0,
                    .layerCount = 1},
            .srcOffset = {.x = 0, .y = 0, .z = 0},
            .dstSubresource = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                    .mipLevel = 0,
                    .baseArrayLayer = 0,
                    .layerCount = 1},
            .dstOffset = {.x = 0, .y = 0, .z = 0},
            //TODO: not hardcoded
            .extent = {.width = 1280,
                    .height = 800,
                    .depth = 1}};

    vkCmdCopyImage(frame_info.command_buffer, rayTraceImageHandle,
                   VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   frame_info.swap_chain_image,
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &imageCopy);

    VkImageMemoryBarrier swapchainPresentMemoryBarrier = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .pNext = NULL,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = 0,
            .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            .srcQueueFamilyIndex = graphics_queue_index,
            .dstQueueFamilyIndex = graphics_queue_index,
            .image = frame_info.swap_chain_image,
            .subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = 1}};

    vkCmdPipelineBarrier(frame_info.command_buffer,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, NULL, 0,
                         NULL, 1, &swapchainPresentMemoryBarrier);

    VkImageMemoryBarrier rayTraceWriteMemoryBarrier = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .pNext = NULL,
            .srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
            .dstAccessMask = 0,
            .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            .newLayout = VK_IMAGE_LAYOUT_GENERAL,
            .srcQueueFamilyIndex = graphics_queue_index,
            .dstQueueFamilyIndex = graphics_queue_index,
            .image = rayTraceImageHandle,
            .subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = 1}};

    vkCmdPipelineBarrier(frame_info.command_buffer,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, NULL, 0,
                         NULL, 1, &rayTraceWriteMemoryBarrier);

    ++current_number_of_frames;
}

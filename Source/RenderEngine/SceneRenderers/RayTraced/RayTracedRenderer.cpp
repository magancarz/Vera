#include "RayTracedRenderer.h"
#include "RenderEngine/RenderingAPI/VulkanHelper.h"

RayTracedRenderer::RayTracedRenderer(Device& device, World* world)
    : device{device}, world{world}
{
    queryRayTracingPipelineProperties();
    createAccelerationStructure();
    createRayTracedImage();
    createCameraUniformBuffer();
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
    ray_tracing_builder.setup(device.findPhysicalQueueFamilies().graphicsFamily, ray_tracing_properties);

    std::vector<RayTracingBuilder::BlasInput> all_blas_inputs;
    all_blas_inputs.reserve(world->objects.size());
    for (auto& [id, object] : world->objects)
    {
        if (object->model)
        {
            RayTracingBuilder::BlasInput blas_input = object->model->getBlasInput();
            all_blas_inputs.emplace_back(blas_input);
        }
    }

    ray_tracing_builder.buildBlas(device, all_blas_inputs[0]);
    ray_tracing_builder.buildTlas(device);
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
//            .addBinding(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR)
//            .addBinding(4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR)
            .build();

    material_descriptor_set_layout = DescriptorSetLayout::Builder(device)
            .addBinding(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR)
            .addBinding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR)
            .build();

    VkWriteDescriptorSetAccelerationStructureKHR
            accelerationStructureDescriptorInfo = {
            .sType =
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
            .pNext = NULL,
            .accelerationStructureCount = 1,
            .pAccelerationStructures = &ray_tracing_builder.topLevelAccelerationStructureHandle};

    auto camera_buffer_descriptor_info = camera_uniform_buffer->descriptorInfo();
    auto vertex_buffer_descriptor_info = world->objects[1]->model->vertex_buffer->descriptorInfo();
    auto index_buffer_descriptor_info = world->objects[1]->model->index_buffer->descriptorInfo();

    VkDescriptorImageInfo rayTraceImageDescriptorInfo = {
            .sampler = VK_NULL_HANDLE,
            .imageView = rayTraceImageViewHandle,
            .imageLayout = VK_IMAGE_LAYOUT_GENERAL};

    DescriptorWriter(*descriptor_set_layout, *descriptor_pool)
            .writeAccelerationStructure(0, &accelerationStructureDescriptorInfo)
            .writeImage(1, &rayTraceImageDescriptorInfo)
            .writeBuffer(2, &camera_buffer_descriptor_info)
//            .writeBuffer(3, &index_buffer_descriptor_info)
//            .writeBuffer(4, &vertex_buffer_descriptor_info)
            .build(descriptor_set_handle);
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
    ray_tracing_pipeline->bindDescriptorSets(frame_info.command_buffer, {descriptor_set_handle});

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
}
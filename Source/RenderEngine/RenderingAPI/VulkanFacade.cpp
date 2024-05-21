#include "VulkanFacade.h"
#include "VulkanHelper.h"

#include <cstring>
#include <iostream>
#include <set>
#include <unordered_set>



VulkanFacade::VulkanFacade(Window& window)
    : window{window}
{
    createSurface();
    pickPhysicalDevice();
    createLogicalDevice();
    createCommandPool();

    VulkanHelper::loadExtensionsFunctions(device);
}

VulkanFacade::~VulkanFacade()
{
    vkDestroyCommandPool(device, command_pool, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroySurfaceKHR(instance.getInstance(), surface_khr, nullptr);
}

void VulkanFacade::pickPhysicalDevice()
{
    uint32_t device_count = 0;
    vkEnumeratePhysicalDevices(instance.getInstance(), &device_count, nullptr);
    if (device_count == 0)
    {
        throw std::runtime_error("Failed to find GPUs with Vulkan support!");
    }
    std::cout << "VulkanFacade count: " << device_count << std::endl;
    std::vector<VkPhysicalDevice> available_physical_devices(device_count);
    vkEnumeratePhysicalDevices(instance.getInstance(), &device_count, available_physical_devices.data());

    for (const auto& physical_device : available_physical_devices)
    {
        if (isDeviceSuitable(physical_device))
        {
            used_physical_device = physical_device;
            break;
        }
    }

    if (used_physical_device == VK_NULL_HANDLE)
    {
        throw std::runtime_error("Failed to find a suitable GPU!");
    }

    vkGetPhysicalDeviceProperties(used_physical_device, &properties);
    std::cout << "Physical device: " << properties.deviceName << std::endl;
}

void VulkanFacade::createLogicalDevice()
{
    QueueFamilyIndices indices = findQueueFamilies(used_physical_device);

    std::vector<VkDeviceQueueCreateInfo> queue_create_infos;
    std::set<uint32_t> unique_queue_families = {indices.graphicsFamily, indices.presentFamily};

    float queue_priority = 1.0f;
    for (uint32_t queue_family: unique_queue_families)
    {
        VkDeviceQueueCreateInfo queue_create_info{};
        queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queue_create_info.queueFamilyIndex = queue_family;
        queue_create_info.queueCount = 1;
        queue_create_info.pQueuePriorities = &queue_priority;
        queue_create_infos.push_back(queue_create_info);
    }

    VkPhysicalDeviceVulkan12Features vulkan_12_features{};
    vulkan_12_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    vulkan_12_features.hostQueryReset = VK_TRUE;
    vulkan_12_features.bufferDeviceAddress = VK_TRUE;
    vulkan_12_features.bufferDeviceAddressCaptureReplay = VK_FALSE;
    vulkan_12_features.bufferDeviceAddressMultiDevice = VK_FALSE;
    vulkan_12_features.descriptorIndexing = VK_TRUE;

    VkPhysicalDeviceAccelerationStructureFeaturesKHR physical_device_acceleration_structure_features{};
    physical_device_acceleration_structure_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
    physical_device_acceleration_structure_features.pNext = &vulkan_12_features;
    physical_device_acceleration_structure_features.accelerationStructure = VK_TRUE;
    physical_device_acceleration_structure_features.accelerationStructureCaptureReplay = VK_FALSE;
    physical_device_acceleration_structure_features.accelerationStructureIndirectBuild = VK_FALSE;
    physical_device_acceleration_structure_features.accelerationStructureHostCommands = VK_FALSE;
    physical_device_acceleration_structure_features.descriptorBindingAccelerationStructureUpdateAfterBind = VK_FALSE;

    VkPhysicalDeviceRayTracingPipelineFeaturesKHR physical_device_ray_tracing_pipeline_features{};
    physical_device_ray_tracing_pipeline_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
    physical_device_ray_tracing_pipeline_features.pNext = &physical_device_acceleration_structure_features;
    physical_device_ray_tracing_pipeline_features.rayTracingPipeline = VK_TRUE;
    physical_device_ray_tracing_pipeline_features.rayTracingPipelineShaderGroupHandleCaptureReplay = VK_FALSE;
    physical_device_ray_tracing_pipeline_features.rayTracingPipelineShaderGroupHandleCaptureReplayMixed = VK_FALSE;
    physical_device_ray_tracing_pipeline_features.rayTracingPipelineTraceRaysIndirect = VK_FALSE;
    physical_device_ray_tracing_pipeline_features.rayTraversalPrimitiveCulling = VK_FALSE;

    VkPhysicalDeviceFeatures device_features{};
    device_features.samplerAnisotropy = VK_TRUE;
    device_features.shaderInt64 = VK_TRUE;

    VkDeviceCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

    create_info.queueCreateInfoCount = static_cast<uint32_t>(queue_create_infos.size());
    create_info.pQueueCreateInfos = queue_create_infos.data();

    create_info.pEnabledFeatures = &device_features;
    create_info.enabledExtensionCount = static_cast<uint32_t>(device_extensions.size());
    create_info.ppEnabledExtensionNames = device_extensions.data();

    create_info.pNext = &physical_device_ray_tracing_pipeline_features;

    auto validation_layers = instance.getEnabledValidationLayers();
    if (instance.validationLayersEnabled())
    {
        create_info.enabledLayerCount = static_cast<uint32_t>(validation_layers.size());
        create_info.ppEnabledLayerNames = validation_layers.data();
    }
    else
    {
        create_info.enabledLayerCount = 0;
    }

    if (vkCreateDevice(used_physical_device, &create_info, nullptr, &device) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create logical getDevice!");
    }

    vkGetDeviceQueue(device, indices.graphicsFamily, 0, &graphics_queue);
    vkGetDeviceQueue(device, indices.presentFamily, 0, &present_queue);
}

void VulkanFacade::createCommandPool()
{
    QueueFamilyIndices queue_family_indices = findPhysicalQueueFamilies();

    VkCommandPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.queueFamilyIndex = queue_family_indices.graphicsFamily;
    pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    if (vkCreateCommandPool(device, &pool_info, nullptr, &command_pool) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create command pool!");
    }
}

void VulkanFacade::createSurface()
{
    window.createWindowSurface(instance.getInstance(), &surface_khr);
}

bool VulkanFacade::isDeviceSuitable(VkPhysicalDevice device)
{
    QueueFamilyIndices indices = findQueueFamilies(device);

    bool extensions_supported = checkDeviceExtensionSupport(device);

    bool swap_chain_adequate;
    if (extensions_supported)
    {
        SwapChainSupportDetails swap_chain_support = querySwapChainSupport(device);
        swap_chain_adequate = !swap_chain_support.formats.empty() && !swap_chain_support.presentModes.empty();
    }

    VkPhysicalDeviceFeatures supported_features;
    vkGetPhysicalDeviceFeatures(device, &supported_features);

    return indices.isComplete() && extensions_supported && swap_chain_adequate && supported_features.samplerAnisotropy;
}

bool VulkanFacade::checkDeviceExtensionSupport(VkPhysicalDevice device)
{
    uint32_t extension_count;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_count, nullptr);

    std::vector<VkExtensionProperties> available_extensions(extension_count);
    vkEnumerateDeviceExtensionProperties(
            device,
            nullptr,
            &extension_count,
            available_extensions.data());

    std::set<std::string> required_extensions(device_extensions.begin(), device_extensions.end());

    for (const auto& extension: available_extensions)
    {
        required_extensions.erase(extension.extensionName);
    }

    return required_extensions.empty();
}

QueueFamilyIndices VulkanFacade::findQueueFamilies(VkPhysicalDevice device)
{
    QueueFamilyIndices indices;

    uint32_t queue_family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, nullptr);

    std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, queue_families.data());

    int i = 0;
    for (const auto& queue_family: queue_families)
    {
        if (queue_family.queueCount > 0 && queue_family.queueFlags & VK_QUEUE_GRAPHICS_BIT)
        {
            indices.graphicsFamily = i;
            indices.graphics_family_has_value = true;
        }
        VkBool32 present_support = false;
        vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface_khr, &present_support);
        if (queue_family.queueCount > 0 && present_support)
        {
            indices.presentFamily = i;
            indices.present_family_has_value = true;
        }
        if (indices.isComplete())
        {
            break;
        }

        ++i;
    }

    return indices;
}

SwapChainSupportDetails VulkanFacade::querySwapChainSupport(VkPhysicalDevice device)
{
    SwapChainSupportDetails details;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface_khr, &details.capabilities);

    uint32_t format_count;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface_khr, &format_count, nullptr);

    if (format_count != 0)
    {
        details.formats.resize(format_count);
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface_khr, &format_count, details.formats.data());
    }

    uint32_t present_mode_count;
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface_khr, &present_mode_count, nullptr);

    if (present_mode_count != 0)
    {
        details.presentModes.resize(present_mode_count);
        vkGetPhysicalDeviceSurfacePresentModesKHR(
                device,
                surface_khr,
                &present_mode_count,
                details.presentModes.data());
    }
    return details;
}

VkFormat VulkanFacade::findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features)
{
    for (VkFormat format : candidates)
    {
        VkFormatProperties props;
        vkGetPhysicalDeviceFormatProperties(used_physical_device, format, &props);

        if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features)
        {
            return format;
        }
        else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features)
        {
            return format;
        }
    }
    throw std::runtime_error("failed to find supported format!");
}

uint32_t VulkanFacade::findMemoryType(uint32_t type_filter, VkMemoryPropertyFlags properties)
{
    VkPhysicalDeviceMemoryProperties memory_properties;
    vkGetPhysicalDeviceMemoryProperties(used_physical_device, &memory_properties);
    for (uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i)
    {
        if ((type_filter & (1 << i)) &&
            (memory_properties.memoryTypes[i].propertyFlags & properties) == properties)
        {
            return i;
        }
    }

    throw std::runtime_error("failed to find suitable memory type!");
}

void VulkanFacade::createBuffer(
        VkDeviceSize size,
        VkBufferUsageFlags usage,
        VkMemoryPropertyFlags properties,
        VkBuffer& buffer,
        VkDeviceMemory& buffer_memory)
{
    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = size;
    buffer_info.usage = usage;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device, &buffer_info, nullptr, &buffer) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create buffer!");
    }

    VkMemoryRequirements memory_requirements;
    vkGetBufferMemoryRequirements(device, buffer, &memory_requirements);

    VkMemoryAllocateFlagsInfo memory_allocate_flags_info{};
    memory_allocate_flags_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
    memory_allocate_flags_info.pNext = nullptr;
    memory_allocate_flags_info.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    memory_allocate_flags_info.deviceMask = 0;

    VkMemoryAllocateInfo allocate_info{};
    allocate_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocate_info.pNext = &memory_allocate_flags_info;
    allocate_info.allocationSize = memory_requirements.size;
    allocate_info.memoryTypeIndex = findMemoryType(memory_requirements.memoryTypeBits, properties);

    if (vkAllocateMemory(device, &allocate_info, nullptr, &buffer_memory) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to allocate buffer memory!");
    }

    vkBindBufferMemory(device, buffer, buffer_memory, 0);
}

void VulkanFacade::createCommandPool(VkCommandPool* command_pool, VkCommandPoolCreateFlags flags)
{
    VkCommandPoolCreateInfo commandPoolCreateInfo{};
    commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    commandPoolCreateInfo.queueFamilyIndex = findQueueFamilies(used_physical_device).graphicsFamily;
    commandPoolCreateInfo.flags = flags;

    if (vkCreateCommandPool(device, &commandPoolCreateInfo, nullptr, command_pool) != VK_SUCCESS)
    {
        throw std::runtime_error("Could not create graphics command pool");
    }
}

void VulkanFacade::createCommandBuffers(VkCommandBuffer* command_buffer, uint32_t command_buffer_count, VkCommandPool& command_pool)
{
    VkCommandBufferAllocateInfo commandBufferAllocateInfo{};
    commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    commandBufferAllocateInfo.commandPool = command_pool;
    commandBufferAllocateInfo.commandBufferCount = command_buffer_count;
    vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, command_buffer);
}

VkCommandBuffer VulkanFacade::beginSingleTimeCommands()
{
    VkCommandBufferAllocateInfo allocate_info{};
    allocate_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocate_info.commandPool = command_pool;
    allocate_info.commandBufferCount = 1;

    VkCommandBuffer command_buffer;
    vkAllocateCommandBuffers(device, &allocate_info, &command_buffer);

    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(command_buffer, &begin_info);
    return command_buffer;
}

void VulkanFacade::endSingleTimeCommands(VkCommandBuffer command_buffer)
{
    vkEndCommandBuffer(command_buffer);

    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &command_buffer;

    vkQueueSubmit(graphics_queue, 1, &submit_info, VK_NULL_HANDLE);
    vkQueueWaitIdle(graphics_queue);

    vkFreeCommandBuffers(device, command_pool, 1, &command_buffer);
}

void VulkanFacade::copyBuffer(VkBuffer src_buffer, VkBuffer dst_buffer, VkDeviceSize size)
{
    VkCommandBuffer command_buffer = beginSingleTimeCommands();

    VkBufferCopy copy_region{};
    copy_region.srcOffset = 0;  // Optional
    copy_region.dstOffset = 0;  // Optional
    copy_region.size = size;
    vkCmdCopyBuffer(command_buffer, src_buffer, dst_buffer, 1, &copy_region);

    endSingleTimeCommands(command_buffer);
}

void VulkanFacade::copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height, uint32_t layer_count)
{
    VkCommandBuffer command_buffer = beginSingleTimeCommands();

    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;

    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = layer_count;

    region.imageOffset = {0, 0, 0};
    region.imageExtent = {width, height, 1};

    vkCmdCopyBufferToImage(
            command_buffer,
            buffer,
            image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1,
            &region);
    endSingleTimeCommands(command_buffer);
}

void VulkanFacade::createImageWithInfo(
        const VkImageCreateInfo& image_info,
        VkMemoryPropertyFlags properties,
        VkImage& image,
        VkDeviceMemory& image_memory)
{
    if (vkCreateImage(device, &image_info, nullptr, &image) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create image!");
    }

    VkMemoryRequirements memory_requirements;
    vkGetImageMemoryRequirements(device, image, &memory_requirements);

    VkMemoryAllocateInfo allocate_info{};
    allocate_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocate_info.allocationSize = memory_requirements.size;
    allocate_info.memoryTypeIndex = findMemoryType(memory_requirements.memoryTypeBits, properties);

    if (vkAllocateMemory(device, &allocate_info, nullptr, &image_memory) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to allocate image memory!");
    }

    if (vkBindImageMemory(device, image, image_memory, 0) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to bind image memory!");
    }
}

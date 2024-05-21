#include "Device.h"

#include <set>

Device::Device(const Instance& instance, const PhysicalDevice& physical_device)
{
    createLogicalDevice(instance, physical_device);
}

Device::~Device()
{
    vkDestroyDevice(device, nullptr);
}

void Device::createLogicalDevice(const Instance& instance, const PhysicalDevice& physical_device)
{
    QueueFamilyIndices indices = physical_device.getQueueFamilyIndices();

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

    auto device_extensions = physical_device.getDeviceExtensions();
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

    if (vkCreateDevice(physical_device.getPhysicalDevice(), &create_info, nullptr, &device) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create logical getDevice!");
    }

    vkGetDeviceQueue(device, indices.graphicsFamily, 0, &graphics_queue);
    vkGetDeviceQueue(device, indices.presentFamily, 0, &present_queue);
}
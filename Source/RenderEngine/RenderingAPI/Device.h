#pragma once

#include <vulkan/vulkan.hpp>

#include <string>
#include <vector>
#include "RenderEngine/Window.h"

struct SwapChainSupportDetails
{
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

struct QueueFamilyIndices
{
    uint32_t graphicsFamily;
    uint32_t presentFamily;
    bool graphics_family_has_value = false;
    bool present_family_has_value = false;

    bool isComplete()
    {
        return graphics_family_has_value && present_family_has_value;
    }
};

class Device
{
public:
#ifdef NDEBUG
    const bool enable_validation_layers = false;
#else
    const bool enable_validation_layers = true;
#endif

    Device(Window& window);
    ~Device();

    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    Device(Device&&) = delete;
    Device& operator=(Device&&) = delete;

    VkCommandPool getCommandPool() { return command_pool; }
    VkDevice getDevice() { return device; }
    VkPhysicalDevice getPhysicalDevice() { return used_physical_device; }
    VkSurfaceKHR surface() { return surface_khr; }
    VkQueue graphicsQueue() { return graphics_queue; }
    VkQueue presentQueue() { return present_queue; }
    SwapChainSupportDetails getSwapChainSupport() { return querySwapChainSupport(used_physical_device); }

    uint32_t findMemoryType(uint32_t type_filter, VkMemoryPropertyFlags properties);
    QueueFamilyIndices findPhysicalQueueFamilies() { return findQueueFamilies(used_physical_device); }
    VkFormat findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features);

    void createBuffer(
        VkDeviceSize size,
        VkBufferUsageFlags usage,
        VkMemoryPropertyFlags properties,
        VkBuffer& buffer,
        VkDeviceMemory& buffer_memory);

    VkCommandBuffer beginSingleTimeCommands();
    void endSingleTimeCommands(VkCommandBuffer command_buffer);

    void copyBuffer(VkBuffer src_buffer, VkBuffer dst_buffer, VkDeviceSize size);
    void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height, uint32_t layer_count);

    void createImageWithInfo(
            const VkImageCreateInfo& image_info,
            VkMemoryPropertyFlags properties,
            VkImage& image,
            VkDeviceMemory& image_memory);

    VkPhysicalDeviceProperties properties;

private:
    void createInstance();
    void setupDebugMessenger();
    bool checkValidationLayerSupport();
    void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& create_info);
    void createSurface();
    void pickPhysicalDevice();
    bool isDeviceSuitable(VkPhysicalDevice device);
    std::vector<const char*> getRequiredExtensions();
    void checkIfInstanceHasGlfwRequiredInstanceExtensions();
    bool checkDeviceExtensionSupport(VkPhysicalDevice device);
    void createLogicalDevice();
    void createCommandPool();
    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device);
    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device);

    Window& window;

    VkInstance instance;
    VkDebugUtilsMessengerEXT debug_messenger;
    VkPhysicalDevice used_physical_device = VK_NULL_HANDLE;
    VkCommandPool command_pool;

    VkDevice device;
    VkSurfaceKHR surface_khr;
    VkQueue graphics_queue;
    VkQueue present_queue;

    const std::vector<const char*> validation_layers = {"VK_LAYER_KHRONOS_validation"};
    const std::vector<const char*> device_extensions =
    {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
        VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
        VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
        VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME,
        VK_KHR_BIND_MEMORY_2_EXTENSION_NAME,
        VK_KHR_SPIRV_1_4_EXTENSION_NAME,
        VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
        VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME
    };
};

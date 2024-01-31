#pragma once

#include <vulkan/vulkan.hpp>

#include <string>
#include <vector>

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

    Device();
    ~Device();
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    Device(Device&&) = delete;
    Device& operator=(Device&&) = delete;

    VkCommandPool getCommandPool() { return command_pool; }
    VkDevice getDevice() { return device; }
    VkSurfaceKHR surface() { return surface_khr; }
    VkQueue graphicsQueue() { return graphics_queue; }
    VkQueue presentQueue() { return present_queue; }
    SwapChainSupportDetails getSwapChainSupport() { return querySwapChainSupport(physical_device); }

    uint32_t findMemoryType(uint32_t type_filter, VkMemoryPropertyFlags properties);
    QueueFamilyIndices findPhysicalQueueFamilies() { return findQueueFamilies(physical_device); }
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
    void hasGLFWRequiredInstanceExtensions();
    bool checkDeviceExtensionSupport(VkPhysicalDevice device);
    void createLogicalDevice();
    void createCommandPool();
    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device);
    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device);

    VkInstance instance;
    VkDebugUtilsMessengerEXT debug_messenger;
    VkPhysicalDevice physical_device = VK_NULL_HANDLE;
    VkCommandPool command_pool;

    VkDevice device;
    VkSurfaceKHR surface_khr;
    VkQueue graphics_queue;
    VkQueue present_queue;

    const std::vector<const char*> validation_layers = {"VK_LAYER_KHRONOS_validation"};
    const std::vector<const char*> device_extensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
};

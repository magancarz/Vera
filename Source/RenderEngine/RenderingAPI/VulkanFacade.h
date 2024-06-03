#pragma once

#include "RenderEngine/Window.h"
#include "Instance.h"
#include "Surface.h"
#include "PhysicalDevice.h"
#include "SwapChainSupportDetails.h"
#include "Device.h"
#include "CommandPool.h"

class VulkanFacade
{
public:
    explicit VulkanFacade(Window& window);

    VulkanFacade(const VulkanFacade&) = delete;
    VulkanFacade& operator=(const VulkanFacade&) = delete;
    VulkanFacade(VulkanFacade&&) = delete;
    VulkanFacade& operator=(VulkanFacade&&) = delete;

    VkInstance getInstance() { return instance.getInstance(); }
    VkDevice getDevice() { return device.getDevice(); }
    VkCommandPool getCommandPool() { return command_pool.getCommandPool(); }
    VkPhysicalDevice getPhysicalDevice() { return used_physical_device.getPhysicalDevice(); }
    VkSurfaceKHR getSurface() { return surface.getSurface(); }
    VkQueue graphicsQueue() { return device.getGraphicsQueue(); }
    VkQueue presentQueue() { return device.getPresentQueue(); }
    SwapChainSupportDetails getSwapChainSupport() { return used_physical_device.querySwapChainSupport(used_physical_device.getPhysicalDevice()); }
    QueueFamilyIndices findPhysicalQueueFamilies() { return used_physical_device.getQueueFamilyIndices(); }

    uint32_t findMemoryType(uint32_t type_filter, VkMemoryPropertyFlags properties);

    VkCommandBuffer beginSingleTimeCommands();
    void endSingleTimeCommands(VkCommandBuffer command_buffer);


    void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height, uint32_t layer_count);

    void createImageWithInfo(
            const VkImageCreateInfo& image_info,
            VkMemoryPropertyFlags properties,
            VkImage& image,
            VkDeviceMemory& image_memory);

private:
    Instance instance;
    Surface surface{instance};
    PhysicalDevice used_physical_device{instance, surface};
    Device device{instance, used_physical_device};
    CommandPool command_pool{device, used_physical_device};

    Window& window;
};

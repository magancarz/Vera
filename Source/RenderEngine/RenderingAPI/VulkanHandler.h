#pragma once

#include "Instance.h"
#include "Surface.h"
#include "PhysicalDevice.h"
#include "Device.h"
#include "CommandPool.h"

class VulkanHandler
{
public:
    VulkanHandler();

    VulkanHandler(const VulkanHandler&) = delete;
    VulkanHandler& operator=(const VulkanHandler&) = delete;
    VulkanHandler(VulkanHandler&&) = delete;
    VulkanHandler& operator=(VulkanHandler&&) = delete;

    [[nodiscard]] Instance& getInstance() { return instance; }
    [[nodiscard]] Surface& getSurface() { return surface; }
    [[nodiscard]] PhysicalDevice& getPhysicalDevice() { return used_physical_device; }
    [[nodiscard]] Device& getLogicalDevice() { return device; }
    [[nodiscard]] CommandPool& getCommandPool() { return command_pool; }

    [[nodiscard]] VkInstance getInstanceHandle() { return instance.getInstance(); }
    [[nodiscard]] VkSurfaceKHR getSurfaceKHRHandle() const { return surface.getSurface(); }
    [[nodiscard]] VkPhysicalDevice getPhysicalDeviceHandle() const { return used_physical_device.getPhysicalDevice(); }
    [[nodiscard]] VkDevice getDeviceHandle() const { return device.getDevice(); }
    [[nodiscard]] VkCommandPool getCommandPoolHandle() const { return command_pool.getCommandPool(); }

private:
    Instance instance;
    Surface surface{instance};
    PhysicalDevice used_physical_device{instance, surface};
    Device device{instance, used_physical_device};
    CommandPool command_pool{device, used_physical_device};
};

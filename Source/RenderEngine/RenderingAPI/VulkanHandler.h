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

    Instance& getInstance() { return instance; }
    Surface& getSurface() { return surface; }
    PhysicalDevice& getPhysicalDevice() { return used_physical_device; }
    Device& getLogicalDevice() { return device; }
    CommandPool& getCommandPool() { return command_pool; }

    VkInstance getInstanceHandle() { return instance.getInstance(); }
    VkSurfaceKHR getSurfaceKHRHandle() { return surface.getSurface(); }
    VkPhysicalDevice getPhysicalDeviceHandle() { return used_physical_device.getPhysicalDevice(); }
    VkDevice getDeviceHandle() { return device.getDevice(); }
    VkCommandPool getCommandPoolHandle() { return command_pool.getCommandPool(); }

private:
    Instance instance;
    Surface surface{instance};
    PhysicalDevice used_physical_device{instance, surface};
    Device device{instance, used_physical_device};
    CommandPool command_pool{device, used_physical_device};
};

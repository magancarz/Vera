#pragma once

#include "Device.h"

class CommandPool
{
public:
    explicit CommandPool(Device& device, PhysicalDevice& physical_device);
    ~CommandPool();

    [[nodiscard]] VkCommandPool getCommandPool() const { return command_pool; }

    void createCommandPool(VkCommandPool* new_command_pool, VkCommandPoolCreateFlags flags);

private:
    Device& device;
    PhysicalDevice& physical_device;

    void createCommandPool();

    VkCommandPool command_pool{VK_NULL_HANDLE};
};

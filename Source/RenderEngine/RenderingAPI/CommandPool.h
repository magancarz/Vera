#pragma once

#include "Device.h"

class CommandPool
{
public:
    CommandPool(Device& device, PhysicalDevice& physical_device);
    ~CommandPool();

    [[nodiscard]] VkCommandPool getCommandPool() const { return command_pool; }

    [[nodiscard]] VkCommandBuffer beginSingleTimeCommands() const;
    void endSingleTimeCommands(VkCommandBuffer command_buffer);

private:
    Device& device;
    PhysicalDevice& physical_device;

    void createCommandPool();

    VkCommandPool command_pool{VK_NULL_HANDLE};
};

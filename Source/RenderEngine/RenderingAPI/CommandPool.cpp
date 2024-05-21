#include "CommandPool.h"

CommandPool::CommandPool(Device& device, PhysicalDevice& physical_device)
    : device{device}, physical_device{physical_device}
{
    createCommandPool();
}

void CommandPool::createCommandPool()
{
    QueueFamilyIndices queue_family_indices = physical_device.getQueueFamilyIndices();

    VkCommandPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.queueFamilyIndex = queue_family_indices.graphicsFamily;
    pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    if (vkCreateCommandPool(device.getDevice(), &pool_info, nullptr, &command_pool) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create command pool!");
    }
}

void CommandPool::createCommandPool(VkCommandPool* new_command_pool, VkCommandPoolCreateFlags flags)
{
    VkCommandPoolCreateInfo commandPoolCreateInfo{};
    commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    commandPoolCreateInfo.queueFamilyIndex = physical_device.getQueueFamilyIndices().graphicsFamily;
    commandPoolCreateInfo.flags = flags;

    if (vkCreateCommandPool(device.getDevice(), &commandPoolCreateInfo, nullptr, new_command_pool) != VK_SUCCESS)
    {
        throw std::runtime_error("Could not create graphics command pool");
    }
}

CommandPool::~CommandPool()
{
    vkDestroyCommandPool(device.getDevice(), command_pool, nullptr);
}
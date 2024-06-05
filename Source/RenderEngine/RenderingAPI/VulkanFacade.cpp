#include "VulkanFacade.h"
#include "VulkanHelper.h"
#include "SwapChainSupportDetails.h"

#include <cstring>
#include <iostream>
#include <set>
#include <unordered_set>

VulkanFacade::VulkanFacade(Window& window)
    : window{window}
{
    VulkanHelper::loadExtensionsFunctions(device.getDevice());
}

uint32_t VulkanFacade::findMemoryType(uint32_t type_filter, VkMemoryPropertyFlags properties)
{
    VkPhysicalDeviceMemoryProperties memory_properties;
    vkGetPhysicalDeviceMemoryProperties(used_physical_device.getPhysicalDevice(), &memory_properties);
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

VkCommandBuffer VulkanFacade::beginSingleTimeCommands()
{
    VkCommandBufferAllocateInfo allocate_info{};
    allocate_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocate_info.commandPool = command_pool.getCommandPool();
    allocate_info.commandBufferCount = 1;

    VkCommandBuffer command_buffer;
    vkAllocateCommandBuffers(device.getDevice(), &allocate_info, &command_buffer);

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

    vkQueueSubmit(device.getGraphicsQueue(), 1, &submit_info, VK_NULL_HANDLE);
    vkQueueWaitIdle(device.getGraphicsQueue());

    vkFreeCommandBuffers(device.getDevice(), command_pool.getCommandPool(), 1, &command_buffer);
}

void VulkanFacade::createImageWithInfo(
        const VkImageCreateInfo& image_info,
        VkMemoryPropertyFlags properties,
        VkImage& image,
        VkDeviceMemory& image_memory)
{
    if (vkCreateImage(device.getDevice(), &image_info, nullptr, &image) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create image!");
    }

    VkMemoryRequirements memory_requirements;
    vkGetImageMemoryRequirements(device.getDevice(), image, &memory_requirements);

    VkMemoryAllocateInfo allocate_info{};
    allocate_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocate_info.allocationSize = memory_requirements.size;
    allocate_info.memoryTypeIndex = findMemoryType(memory_requirements.memoryTypeBits, properties);

    if (vkAllocateMemory(device.getDevice(), &allocate_info, nullptr, &image_memory) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to allocate image memory!");
    }

    if (vkBindImageMemory(device.getDevice(), image, image_memory, 0) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to bind image memory!");
    }
}

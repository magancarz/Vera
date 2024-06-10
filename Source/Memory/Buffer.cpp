#include "Buffer.h"

#include <cassert>
#include <cstring>

#include <vulkan/vulkan_core.h>

#include "Logs/LogSystem.h"

Buffer::Buffer(
    Device& logical_device,
    CommandPool& command_pool,
    const VulkanMemoryAllocatorInfo& allocator_info,
    VkBuffer buffer,
    uint32_t buffer_size)
    : logical_device{logical_device}, command_pool{command_pool}, allocator_info{allocator_info}, buffer{buffer},
      buffer_size{buffer_size}
{
}

Buffer::~Buffer()
{
    unmap();
    vmaDestroyBuffer(allocator_info.vma_allocator, buffer, allocator_info.vma_allocation);
}

VkResult Buffer::map()
{
    return vmaMapMemory(allocator_info.vma_allocator, allocator_info.vma_allocation, &mapped);
}

void Buffer::unmap()
{
    if (mapped)
    {
        vmaUnmapMemory(allocator_info.vma_allocator, allocator_info.vma_allocation);
        mapped = nullptr;
    }
}

void Buffer::writeToBuffer(const void* data)
{
    if (!mapped)
    {
        LogSystem::log(LogSeverity::WARNING, "Tried to copy to unmapped buffer! Mapping...");
        map();
    }
    memcpy(mapped, data, buffer_size);
}

void Buffer::copyFrom(const Buffer& src_buffer) const
{
    VkCommandBuffer command_buffer = command_pool.beginSingleTimeCommands();

    VkBufferCopy copy_region{};
    copy_region.size = src_buffer.buffer_size;
    vkCmdCopyBuffer(command_buffer, src_buffer.getBuffer(), buffer, 1, &copy_region);

    command_pool.endSingleTimeCommands(command_buffer);
}

VkDescriptorBufferInfo Buffer::descriptorInfo() const
{
    return VkDescriptorBufferInfo
    {
        buffer,
        0,
        VK_WHOLE_SIZE,
    };
}

VkDeviceAddress Buffer::getBufferDeviceAddress() const
{
    if (buffer == VK_NULL_HANDLE)
    {
        return 0;
    }

    VkBufferDeviceAddressInfo info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    info.buffer = buffer;

    return vkGetBufferDeviceAddress(logical_device.getDevice(), &info);
}

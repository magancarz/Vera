#include "Buffer.h"

#include <cassert>
#include <cstring>

#include <vulkan/vulkan_core.h>

Buffer::Buffer(VulkanFacade& vulkan_facade, const VulkanMemoryAllocatorInfo& allocator_info, VkBuffer buffer, uint32_t buffer_size)
    : vulkan_facade{vulkan_facade}, allocator_info{allocator_info}, buffer{buffer}, buffer_size{buffer_size} {}

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

void Buffer::writeToBuffer(void* data)
{
    assert(mapped && "Cannot copy to unmapped buffer");
    memcpy(mapped, data, buffer_size);
}

void Buffer::copyFromBuffer(const std::unique_ptr<Buffer>& src_buffer)
{
    VkCommandBuffer command_buffer = vulkan_facade.beginSingleTimeCommands();

    VkBufferCopy copy_region{};
    copy_region.size = src_buffer->buffer_size;
    vkCmdCopyBuffer(command_buffer, src_buffer->getBuffer(), buffer, 1, &copy_region);

    vulkan_facade.endSingleTimeCommands(command_buffer);
}

VkResult Buffer::flush() const
{
    return vmaFlushAllocation(allocator_info.vma_allocator, allocator_info.vma_allocation, 0, VK_WHOLE_SIZE);
}

VkDescriptorBufferInfo Buffer::descriptorInfo()
{
    return VkDescriptorBufferInfo
    {
        buffer,
        0,
        VK_WHOLE_SIZE,
    };
}

VkDeviceAddress Buffer::getBufferDeviceAddress()
{
    if(buffer == VK_NULL_HANDLE)
    {
        return 0ULL;
    }

    VkBufferDeviceAddressInfo info = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
            .buffer = buffer};

    return vkGetBufferDeviceAddress(vulkan_facade.getDevice(), &info);
}
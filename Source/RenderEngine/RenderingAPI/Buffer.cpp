#include "Buffer.h"

#include <cassert>
#include <cstring>

#include <vulkan/vulkan_core.h>

VkDeviceSize Buffer::getAlignment(VkDeviceSize instance_size, VkDeviceSize min_offset_alignment)
{
    if (min_offset_alignment > 0)
    {
        return (instance_size + min_offset_alignment - 1) & ~(min_offset_alignment - 1);
    }
    return instance_size;
}

Buffer::Buffer(
        Device& device,
        VkDeviceSize instance_size,
        uint32_t instance_count,
        VkBufferUsageFlags usage_flags,
        VkMemoryPropertyFlags memory_property_flags,
        VkDeviceSize min_offset_alignment)
        : device{device},
          instance_size{instance_size},
          instance_count{instance_count},
          usage_flags{usage_flags},
          memory_property_flags{memory_property_flags}
{
    alignment_size = getAlignment(instance_size, min_offset_alignment);
    buffer_size = alignment_size * instance_count;
    device.createBuffer(buffer_size, usage_flags, memory_property_flags, buffer, memory);
}

Buffer::~Buffer()
{
    unmap();
    vkDestroyBuffer(device.getDevice(), buffer, nullptr);
    vkFreeMemory(device.getDevice(), memory, nullptr);
}

VkResult Buffer::map(VkDeviceSize size, VkDeviceSize offset)
{
    assert(buffer && memory && "Called map on buffer before create");
    return vkMapMemory(device.getDevice(), memory, offset, size, 0, &mapped);
}

void Buffer::unmap()
{
    if (mapped)
    {
        vkUnmapMemory(device.getDevice(), memory);
        mapped = nullptr;
    }
}

void Buffer::writeToBuffer(void* data, VkDeviceSize size, VkDeviceSize offset)
{
    assert(mapped && "Cannot copy to unmapped buffer");

    if (size == VK_WHOLE_SIZE)
    {
        memcpy(mapped, data, buffer_size);
    }
    else
    {
        char* mem_offset = (char*)mapped;
        mem_offset += offset;
        memcpy(mem_offset, data, size);
    }
}

void Buffer::writeWithStagingBuffer(void* data)
{
    Buffer staging_buffer
    {
            device,
            instance_size,
            instance_count,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    };

    staging_buffer.map();
    staging_buffer.writeToBuffer(data);
    device.copyBuffer(staging_buffer.getBuffer(), getBuffer(), buffer_size);
}

VkResult Buffer::flush(VkDeviceSize size, VkDeviceSize offset)
{
    VkMappedMemoryRange mapped_range{};
    mapped_range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    mapped_range.memory = memory;
    mapped_range.offset = offset;
    mapped_range.size = size;
    return vkFlushMappedMemoryRanges(device.getDevice(), 1, &mapped_range);
}

VkResult Buffer::invalidate(VkDeviceSize size, VkDeviceSize offset)
{
    VkMappedMemoryRange mapped_range{};
    mapped_range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    mapped_range.memory = memory;
    mapped_range.offset = offset;
    mapped_range.size = size;
    return vkInvalidateMappedMemoryRanges(device.getDevice(), 1, &mapped_range);
}

VkDescriptorBufferInfo Buffer::descriptorInfo(VkDeviceSize size, VkDeviceSize offset)
{
    return VkDescriptorBufferInfo
    {
        buffer,
        offset,
        size,
    };
}

void Buffer::writeToIndex(void *data, int index)
{
    writeToBuffer(data, instance_size, index * alignment_size);
}

VkResult Buffer::flushIndex(int index) { return flush(alignment_size, index * alignment_size); }

VkDescriptorBufferInfo Buffer::descriptorInfoForIndex(int index)
{
    return descriptorInfo(alignment_size, index * alignment_size);
}

VkResult Buffer::invalidateIndex(int index)
{
    return invalidate(alignment_size, index * alignment_size);
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

    return vkGetBufferDeviceAddress(device.getDevice(), &info);
}
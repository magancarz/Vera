#include "Buffer.h"

#include <cassert>
#include <cstring>

#include <vulkan/vulkan_core.h>

/**
 * Returns the minimum instance size required to be compatible with devices minOffsetAlignment
 *
 * @param instance_size The size of an instance
 * @param min_offset_alignment The minimum required alignment, in bytes, for the offset member (eg
 * minUniformBufferOffsetAlignment)
 *
 * @return VkResult of the buffer mapping call
 */
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

/**
* Map a memory range of this buffer. If successful, mapped points to the specified buffer range.
*
* @param size (Optional) Size of the memory range to map. Pass VK_WHOLE_SIZE to map the complete
* buffer range.
* @param offset (Optional) Byte offset from beginning
*
* @return VkResult of the buffer mapping call
*/
VkResult Buffer::map(VkDeviceSize size, VkDeviceSize offset)
{
    assert(buffer && memory && "Called map on buffer before create");
    return vkMapMemory(device.getDevice(), memory, offset, size, 0, &mapped);
}

/**
* Unmap a mapped memory range
*
* @note Does not return a result as vkUnmapMemory can't fail
*/
void Buffer::unmap()
{
    if (mapped)
    {
        vkUnmapMemory(device.getDevice(), memory);
        mapped = nullptr;
    }
}

/**
* Copies the specified data to the mapped buffer. Default value writes whole buffer range
*
* @param data Pointer to the data to copy
* @param size (Optional) Size of the data to copy. Pass VK_WHOLE_SIZE to flush the complete buffer
* range.
* @param offset (Optional) Byte offset from beginning of mapped region
*
*/
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

//TODO: do it with data size and offset
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

/**
* Flush a memory range of the buffer to make it visible to the device
*
* @note Only required for non-coherent memory
*
* @param size (Optional) Size of the memory range to flush. Pass VK_WHOLE_SIZE to flush the
* complete buffer range.
* @param offset (Optional) Byte offset from beginning
*
* @return VkResult of the flush call
*/
VkResult Buffer::flush(VkDeviceSize size, VkDeviceSize offset)
{
    VkMappedMemoryRange mapped_range{};
    mapped_range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    mapped_range.memory = memory;
    mapped_range.offset = offset;
    mapped_range.size = size;
    return vkFlushMappedMemoryRanges(device.getDevice(), 1, &mapped_range);
}

/**
* Invalidate a memory range of the buffer to make it visible to the host
*
* @note Only required for non-coherent memory
*
* @param size (Optional) Size of the memory range to invalidate. Pass VK_WHOLE_SIZE to invalidate
* the complete buffer range.
* @param offset (Optional) Byte offset from beginning
*
* @return VkResult of the invalidate call
*/
VkResult Buffer::invalidate(VkDeviceSize size, VkDeviceSize offset)
{
    VkMappedMemoryRange mapped_range{};
    mapped_range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    mapped_range.memory = memory;
    mapped_range.offset = offset;
    mapped_range.size = size;
    return vkInvalidateMappedMemoryRanges(device.getDevice(), 1, &mapped_range);
}

/**
* Create a buffer info descriptor
*
* @param size (Optional) Size of the memory range of the descriptor
* @param offset (Optional) Byte offset from beginning
*
* @return VkDescriptorBufferInfo of specified offset and range
*/
VkDescriptorBufferInfo Buffer::descriptorInfo(VkDeviceSize size, VkDeviceSize offset)
{
    return VkDescriptorBufferInfo
    {
        buffer,
        offset,
        size,
    };
}

/**
* Copies "instanceSize" bytes of data to the mapped buffer at an offset of index * alignmentSize
*
* @param data Pointer to the data to copy
* @param index Used in offset calculation
*
*/
void Buffer::writeToIndex(void *data, int index)
{
    writeToBuffer(data, instance_size, index * alignment_size);
}

/**
*  Flush the memory range at index * alignmentSize of the buffer to make it visible to the device
*
* @param index Used in offset calculation
*
*/
VkResult Buffer::flushIndex(int index) { return flush(alignment_size, index * alignment_size); }

/**
* Create a buffer info descriptor
*
* @param index Specifies the region given by index * alignmentSize
*
* @return VkDescriptorBufferInfo for instance at index
*/
VkDescriptorBufferInfo Buffer::descriptorInfoForIndex(int index)
{
    return descriptorInfo(alignment_size, index * alignment_size);
}

/**
* Invalidate a memory range of the buffer to make it visible to the host
*
* @note Only required for non-coherent memory
*
* @param index Specifies the region to invalidate: index * alignmentSize
*
* @return VkResult of the invalidate call
*/
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
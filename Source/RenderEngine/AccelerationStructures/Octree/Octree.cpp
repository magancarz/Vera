#include "Octree.h"

#include "glm/vec3.hpp"

#include "Memory/MemoryAllocator.h"
#include "RenderEngine/AccelerationStructures/AABB.h"

Octree::Octree(MemoryAllocator& memory_allocator)
    : memory_allocator{memory_allocator}, aabb_buffer{createAABBBuffer()} {}

std::unique_ptr<Buffer> Octree::createAABBBuffer()
{
    AABB aabb{.min = glm::vec3{-1, -1, -1}, .max = glm::vec3{1, 1, 1}};

    auto staging_buffer = memory_allocator.createStagingBuffer(
        sizeof(AABB),
        1,
        &aabb);

    BufferInfo aabb_buffer_info{};
    aabb_buffer_info.instance_size = sizeof(AABB);
    aabb_buffer_info.instance_count = 1;
    aabb_buffer_info.usage_flags =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
    aabb_buffer_info.required_memory_flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;

    auto aabb_buffer = memory_allocator.createBuffer(aabb_buffer_info);
    aabb_buffer->copyFrom(*staging_buffer);

    return aabb_buffer;
}

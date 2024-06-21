#include "OctreeBlas.h"

#include <RenderEngine/AccelerationStructures/AABB.h>

#include <utility>

OctreeBlas::OctreeBlas(
        VulkanHandler& device,
        MemoryAllocator& memory_allocator,
        Octree octree)
    : Blas{device, memory_allocator}, octree{std::move(octree)} {}

void OctreeBlas::createBlas()
{
    allocateOctreeNodesOnGPUMemory();
    allocateOctreeAABBOnGPUMemory();

    blas_input = BlasBuilder::BlasInput{};

    VkAccelerationStructureGeometryAabbsDataKHR aabbs{};
    aabbs.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR;
    aabbs.data.deviceAddress = aabb_buffer->getBufferDeviceAddress();
    aabbs.stride = sizeof(AABB);

    VkAccelerationStructureGeometryKHR acceleration_structure_geometry{};
    acceleration_structure_geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    acceleration_structure_geometry.geometryType = VK_GEOMETRY_TYPE_AABBS_KHR;
    acceleration_structure_geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    acceleration_structure_geometry.geometry.aabbs = aabbs;

    VkAccelerationStructureBuildRangeInfoKHR offset{};
    offset.firstVertex = 0;
    offset.primitiveCount = 1;
    offset.primitiveOffset = 0;
    offset.transformOffset = 0;

    blas_input.acceleration_structure_geometry.emplace_back(acceleration_structure_geometry);
    blas_input.acceleration_structure_build_offset_info.emplace_back(offset);

    //TODO: currently there are no batch creation of blases so 'blases' list will be always size 1
    std::vector<AccelerationStructure> blases = BlasBuilder::buildBottomLevelAccelerationStructures(
        device, memory_allocator, {blas_input}, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);
    blas = std::move(blases.front());
}

void OctreeBlas::allocateOctreeNodesOnGPUMemory()
{
    auto staging_buffer = memory_allocator.createStagingBuffer(
        sizeof(OctreeNode),
        static_cast<uint32_t>(octree.nodes().size()),
        octree.nodes().data());

    BufferInfo octree_nodes_buffer_info{};
    octree_nodes_buffer_info.instance_size = sizeof(OctreeNode);
    octree_nodes_buffer_info.instance_count = static_cast<uint32_t>(octree.nodes().size());
    octree_nodes_buffer_info.usage_flags =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    octree_nodes_buffer_info.required_memory_flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;

    octree_buffer = memory_allocator.createBuffer(octree_nodes_buffer_info);
    octree_buffer->copyFrom(*staging_buffer);
}

void OctreeBlas::allocateOctreeAABBOnGPUMemory()
{
    AABB octree_aabb = octree.aabb();
    auto staging_buffer = memory_allocator.createStagingBuffer(sizeof(AABB), 1, &octree_aabb);

    BufferInfo aabb_buffer_info{};
    aabb_buffer_info.instance_size = sizeof(AABB);
    aabb_buffer_info.instance_count = 1;
    aabb_buffer_info.usage_flags =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
    aabb_buffer_info.required_memory_flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;

    aabb_buffer = memory_allocator.createBuffer(aabb_buffer_info);
    aabb_buffer->copyFrom(*staging_buffer);
}

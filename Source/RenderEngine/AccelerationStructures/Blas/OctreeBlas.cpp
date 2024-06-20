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
    octree_buffer

    blas_input = BlasBuilder::BlasInput{};

    VkAccelerationStructureGeometryAabbsDataKHR aabbs{};
    aabbs.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR;
    //TODO: remember
    // aabbs.data.deviceAddress = octree.aabbBuffer().getBufferDeviceAddress();
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

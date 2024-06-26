#pragma once

#include <vector>

#include "AccelerationStructure.h"

class MemoryAllocator;
struct BlasInstance;

class Tlas
{
public:
    Tlas(
        Device& logical_device,
        CommandPool& command_pool,
        MemoryAllocator& memory_allocator,
        VkBuildAccelerationStructureFlagsKHR build_flags,
        const std::vector<BlasInstance>& blas_instances);

    Tlas(const Tlas&) = delete;
    Tlas& operator=(const Tlas&) = delete;
    Tlas(Tlas&&) = delete;
    Tlas& operator=(Tlas&&) = delete;

    void update(const std::vector<BlasInstance>& blas_instances);

    [[nodiscard]] const AccelerationStructure& accelerationStructure() const { return acceleration_structure; }

private:
    Device& logical_device;
    CommandPool& command_pool;
    MemoryAllocator& memory_allocator;
    VkBuildAccelerationStructureFlagsKHR build_flags;

    AccelerationStructure build(const std::vector<BlasInstance>& blas_instances);

    void copyBlasInstancesDataToBuffer(const std::vector<BlasInstance>& blas_instances);
    void createBlasInstancesBuffer(uint32_t blas_instances_count);

    std::unique_ptr<Buffer> blas_instances_staging_buffer;
    std::unique_ptr<Buffer> blas_instances_buffer;

    VkAccelerationStructureGeometryKHR fillGeometryInfo();
    VkAccelerationStructureBuildGeometryInfoKHR fillBuildGeometryInfo(
        VkAccelerationStructureGeometryKHR* geometries, VkBuildAccelerationStructureModeKHR mode) const;
    VkAccelerationStructureBuildSizesInfoKHR obtainBuildSizesInfo(
        const VkAccelerationStructureBuildGeometryInfoKHR& build_geometry_info, uint32_t blas_instances_count);

    std::unique_ptr<Buffer> createScratchBuffer(uint32_t scratch_buffer_size);

    AccelerationStructure createAccelerationStructure(uint32_t acceleration_structure_size);

    AccelerationStructure acceleration_structure;

    void cmdBuildTlas(const VkAccelerationStructureBuildGeometryInfoKHR& build_geometry_info, VkAccelerationStructureBuildRangeInfoKHR* build_range_info);
};

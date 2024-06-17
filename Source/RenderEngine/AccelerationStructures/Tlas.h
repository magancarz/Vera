#pragma once

#include <vector>

#include "AccelerationStructure.h"

class MemoryAllocator;
struct BlasInstance;

class Tlas
{
public:
    Tlas(Device& logical_device, CommandPool& command_pool, MemoryAllocator& memory_allocator, VkBuildAccelerationStructureFlagsKHR build_flags);

    void build(const std::vector<BlasInstance>& blas_instances);
	void update(const std::vector<BlasInstance>& blas_instances);

    const AccelerationStructure& accelerationStructure() const { return acceleration_structure; }

private:
	Device& logical_device;
	CommandPool& command_pool;
	MemoryAllocator& memory_allocator;
    VkBuildAccelerationStructureFlagsKHR build_flags;

    void copyBlasInstancesDataToBuffer(const std::vector<BlasInstance>& blas_instances);
    void createBlasInstancesBuffer(uint32_t blas_instances_count);

    std::unique_ptr<Buffer> blas_instances_staging_buffer;
    std::unique_ptr<Buffer> blas_instances_buffer;

    std::pair<VkAccelerationStructureGeometryKHR, VkAccelerationStructureBuildGeometryInfoKHR> fillBuildGeometryInfo(VkBuildAccelerationStructureModeKHR mode);
    VkAccelerationStructureBuildSizesInfoKHR obtainBuildSizesInfo(const VkAccelerationStructureBuildGeometryInfoKHR& build_geometry_info, uint32_t blas_instances_count);

    std::unique_ptr<Buffer> createScratchBuffer(uint32_t scratch_buffer_size);

    void createAccelerationStructure(uint32_t acceleration_structure_size);

    AccelerationStructure acceleration_structure{};

    void cmdBuildTlas(const VkAccelerationStructureBuildGeometryInfoKHR& build_geometry_info, VkAccelerationStructureBuildRangeInfoKHR* build_range_info);
};
#include "Blas.h"
#include "RenderEngine/Models/RayTracingAccelerationStructureBuilder.h"
#include "VulkanHelper.h"
#include "VulkanDefines.h"
#include "Vertex.h"

Blas::Blas(VulkanFacade& device, std::unique_ptr<MemoryAllocator>& memory_allocator, const MeshComponent* mesh_component)
    : device{device}, memory_allocator{memory_allocator}
{
    createBlas(mesh_component);
}

void Blas::createBlas(const MeshComponent* mesh_component)
{
    MeshDescription mesh_description = mesh_component->getDescription();

    RayTracingAccelerationStructureBuilder::BlasInput blas_input;
    for (auto& model_description : mesh_description.model_descriptions)
    {
        VkDeviceAddress vertex_address = model_description.vertex_address;
        VkDeviceAddress index_address = model_description.index_address;

        uint32_t max_primitive_count = model_description.num_of_triangles;

        VkAccelerationStructureGeometryTrianglesDataKHR triangles{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR};
        triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
        triangles.vertexData.deviceAddress = vertex_address;
        triangles.vertexStride = sizeof(Vertex);

        triangles.indexType = VK_INDEX_TYPE_UINT32;
        triangles.indexData.deviceAddress = index_address;

        triangles.maxVertex = model_description.num_of_triangles * 3 - 1;

        VkAccelerationStructureGeometryKHR acceleration_structure_geometry{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
        acceleration_structure_geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
        acceleration_structure_geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
        acceleration_structure_geometry.geometry.triangles = triangles;

        VkAccelerationStructureBuildRangeInfoKHR offset{};
        offset.firstVertex = 0;
        offset.primitiveCount = max_primitive_count;
        offset.primitiveOffset = 0;
        offset.transformOffset = 0;

        blas_input.acceleration_structure_geometry.emplace_back(acceleration_structure_geometry);
        blas_input.acceleration_structure_build_offset_info.emplace_back(offset);
    }
    blas_input.flags = VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR;

    RayTracingAccelerationStructureBuilder builder{device, memory_allocator};
    blas = std::move(builder.buildBottomLevelAccelerationStructures({blas_input}, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR)[0]);
}

Blas::~Blas()
{
    pvkDestroyAccelerationStructureKHR(device.getDevice(), blas.acceleration_structure, VulkanDefines::NO_CALLBACK);
}

BlasInstance Blas::createBlasInstance(const glm::mat4& transform)
{
    BlasInstance blas_instance{};
    blas_instance.bottomLevelAccelerationStructureInstance.transform = VulkanHelper::mat4ToVkTransformMatrixKHR(transform);
    blas_instance.bottomLevelAccelerationStructureInstance.mask = 0xFF;
    blas_instance.bottomLevelAccelerationStructureInstance.instanceShaderBindingTableRecordOffset = 0;
    blas_instance.bottomLevelAccelerationStructureInstance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    blas_instance.bottomLevelAccelerationStructureInstance.accelerationStructureReference = blas.bottom_level_acceleration_structure_device_address;

    blas_instance.bottom_level_geometry_instance_buffer = memory_allocator->createBuffer
    (
            sizeof(VkAccelerationStructureInstanceKHR),
            1,
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
    );
    blas_instance.bottom_level_geometry_instance_buffer->map();
    blas_instance.bottom_level_geometry_instance_buffer->writeToBuffer(&blas_instance.bottomLevelAccelerationStructureInstance);
    blas_instance.bottom_level_geometry_instance_buffer->unmap();

    blas_instance.bottomLevelGeometryInstanceDeviceAddress = blas_instance.bottom_level_geometry_instance_buffer->getBufferDeviceAddress();

    return blas_instance;
}
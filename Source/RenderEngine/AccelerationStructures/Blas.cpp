#include "Blas.h"

#include "BlasBuilder.h"
#include "RenderEngine/RenderingAPI/VulkanHelper.h"
#include "RenderEngine/RenderingAPI/VulkanDefines.h"
#include "Assets/Model/Vertex.h"
#include "Assets/AssetManager.h"
#include "Assets/Mesh.h"
#include "Assets/Model/Model.h"

Blas::Blas(
    VulkanHandler& device,
    MemoryAllocator& memory_allocator,
    AssetManager& asset_manager,
    const Mesh& mesh)
    : device{device}, memory_allocator{memory_allocator}, asset_manager{asset_manager}
{
    createBlas(mesh);
}

void Blas::createBlas(const Mesh& mesh)
{
    blas_input = BlasBuilder::BlasInput{};

    for (auto& model : mesh.models)
    {
        ModelDescription model_description = model->getModelDescription();

        VkDeviceAddress vertex_address = model_description.vertex_buffer->getBufferDeviceAddress();
        VkDeviceAddress index_address = model_description.index_buffer->getBufferDeviceAddress();

        auto max_primitive_count = static_cast<uint32_t>(model_description.num_of_triangles);

        VkAccelerationStructureGeometryTrianglesDataKHR triangles{};
        triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
        triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
        triangles.vertexData.deviceAddress = vertex_address;
        triangles.vertexStride = sizeof(Vertex);

        triangles.indexType = VK_INDEX_TYPE_UINT32;
        triangles.indexData.deviceAddress = index_address;

        triangles.maxVertex = static_cast<uint32_t>(model_description.num_of_triangles * 3 - 1);

        VkAccelerationStructureGeometryKHR acceleration_structure_geometry{};
        acceleration_structure_geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
        acceleration_structure_geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;

        auto material = asset_manager.fetchMaterial(model->getRequiredMaterial());
        acceleration_structure_geometry.flags = material->isOpaque() ? VK_GEOMETRY_OPAQUE_BIT_KHR : VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR;
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
    blas = std::move(BlasBuilder::buildBottomLevelAccelerationStructures(
        device, memory_allocator, {blas_input}, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR).front());
}

Blas::~Blas()
{
    pvkDestroyAccelerationStructureKHR(device.getDeviceHandle(), blas.handle, VulkanDefines::NO_CALLBACK);
}

BlasInstance Blas::createBlasInstance(const glm::mat4& transform) const
{
    BlasInstance blas_instance{};
    blas_instance.bottom_level_acceleration_structure_instance.transform = VulkanHelper::mat4ToVkTransformMatrixKHR(transform);
    blas_instance.bottom_level_acceleration_structure_instance.mask = 0xFF;
    blas_instance.bottom_level_acceleration_structure_instance.instanceShaderBindingTableRecordOffset = 0;
    blas_instance.bottom_level_acceleration_structure_instance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    blas_instance.bottom_level_acceleration_structure_instance.accelerationStructureReference = blas.buffer->getBufferDeviceAddress();

    BufferInfo bottom_level_geometry_instance_buffer_info{};
    bottom_level_geometry_instance_buffer_info.instance_size = sizeof(VkAccelerationStructureInstanceKHR);
    bottom_level_geometry_instance_buffer_info.instance_count = 1;
    bottom_level_geometry_instance_buffer_info.usage_flags =
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    bottom_level_geometry_instance_buffer_info.required_memory_flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    bottom_level_geometry_instance_buffer_info.allocation_flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

    blas_instance.bottom_level_geometry_instance_buffer = memory_allocator.createBuffer(bottom_level_geometry_instance_buffer_info);
    blas_instance.bottom_level_geometry_instance_buffer->map();
    blas_instance.bottom_level_geometry_instance_buffer->writeToBuffer(&blas_instance.bottom_level_acceleration_structure_instance);
    blas_instance.bottom_level_geometry_instance_buffer->unmap();

    blas_instance.bottom_level_geometry_instance_device_address = blas_instance.bottom_level_geometry_instance_buffer->getBufferDeviceAddress();

    return blas_instance;
}

void Blas::update()
{
    BlasBuilder::updateBottomLevelAccelerationStructures(
        device, memory_allocator, {blas.handle}, {blas_input}, VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);
}

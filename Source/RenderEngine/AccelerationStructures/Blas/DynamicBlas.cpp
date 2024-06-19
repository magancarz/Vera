#include "DynamicBlas.h"

#include "Assets/AssetManager.h"

DynamicBlas::DynamicBlas(VulkanHandler& device, MemoryAllocator& memory_allocator, AssetManager& asset_manager, const Mesh& mesh)
    : Blas(device, memory_allocator), asset_manager{asset_manager}, mesh{mesh} {}

void DynamicBlas::createBlas()
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

    //TODO: currently there are no batch creation of blases so 'blases' list will be always size 1
    std::vector<AccelerationStructure> blases = BlasBuilder::buildBottomLevelAccelerationStructures(
        device, memory_allocator, {blas_input}, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);
    blas = std::move(blases.front());
}
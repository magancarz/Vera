#include "Model.h"

#include "RenderEngine/RenderingAPI/VulkanHelper.h"
#include "Memory/MemoryAllocator.h"
#include "ModelInfo.h"

Model::Model(
        MemoryAllocator& memory_allocator,
        const ModelInfo& model_info)
    : name{model_info.name}, required_material{model_info.required_material}
{
    createVertexBuffer(memory_allocator, model_info.vertices);
    createIndexBuffer(memory_allocator, model_info.indices);
}

void Model::createVertexBuffer(MemoryAllocator& memory_allocator, const std::vector<Vertex>& vertices)
{
    vertex_count = static_cast<uint32_t>(vertices.size());
    assert(vertex_count >= 3 && "Vertex count must be at least 3.");

    auto staging_buffer = memory_allocator.createStagingBuffer(
        sizeof(Vertex),
        static_cast<uint32_t>(vertices.size()),
        vertices.data());
    vertex_buffer = memory_allocator.createBuffer(
        sizeof(Vertex),
        static_cast<uint32_t>(vertices.size()),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT);
    vertex_buffer->copyFromBuffer(*staging_buffer);
}

void Model::createIndexBuffer(MemoryAllocator& memory_allocator, const std::vector<uint32_t>& indices)
{
    index_count = static_cast<uint32_t>(indices.size());

    auto staging_buffer = memory_allocator.createStagingBuffer(
        sizeof(uint32_t),
        static_cast<uint32_t>(indices.size()),
        indices.data());
    index_buffer = memory_allocator.createBuffer(
        sizeof(uint32_t),
        static_cast<uint32_t>(indices.size()),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT);
    index_buffer->copyFromBuffer(*staging_buffer);
}

ModelDescription Model::getModelDescription() const
{
    assert(vertex_buffer && index_buffer && index_count >= 3 && "Model should be valid!");

    ModelDescription model_description{};
    model_description.vertex_buffer = vertex_buffer.get();
    model_description.index_buffer = index_buffer.get();
    model_description.num_of_triangles = index_count / 3;
    model_description.num_of_vertices = vertex_count;
    model_description.num_of_indices = index_count;

    return model_description;
}

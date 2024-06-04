#include "OBJModel.h"

#include "Utils/Algorithms.h"

OBJModel::OBJModel(MemoryAllocator& memory_allocator, const OBJModelInfo& obj_model_info)
        : Model(obj_model_info.name, obj_model_info.required_material)
{
    createVertexBuffer(memory_allocator, obj_model_info.vertices);
    createIndexBuffer(memory_allocator, obj_model_info.indices);
}

void OBJModel::createVertexBuffer(MemoryAllocator& memory_allocator, const std::vector<Vertex>& vertices)
{
    vertex_count = static_cast<uint32_t>(vertices.size());
    assert(vertex_count >= 3 && "Vertex count must be at least 3.");

    auto staging_buffer = memory_allocator.createStagingBuffer(sizeof(Vertex), vertices.size(), (void*)vertices.data());
    vertex_buffer = memory_allocator.createBuffer(
            sizeof(Vertex),
            vertices.size(),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT |
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
            VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT);
    vertex_buffer->copyFromBuffer(staging_buffer);
}

void OBJModel::createIndexBuffer(MemoryAllocator& memory_allocator, const std::vector<uint32_t>& indices)
{
    index_count = static_cast<uint32_t>(indices.size());

    auto staging_buffer = memory_allocator.createStagingBuffer(sizeof(uint32_t), indices.size(), (void*)indices.data());
    index_buffer = memory_allocator.createBuffer(
            sizeof(uint32_t),
            indices.size(),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT |
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
            VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT);
    index_buffer->copyFromBuffer(staging_buffer);
}

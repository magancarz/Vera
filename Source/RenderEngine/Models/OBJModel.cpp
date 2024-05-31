#include "OBJModel.h"

#include <unordered_map>
#include <iostream>

#include "Utils/Algorithms.h"

OBJModel::OBJModel(const std::unique_ptr<MemoryAllocator>& memory_allocator, const std::vector<OBJModelInfo>& obj_models_info, std::string name)
    : Model(std::move(name))
{
    createManyModels(memory_allocator, obj_models_info);
}

void OBJModel::createManyModels(const std::unique_ptr<MemoryAllocator>& memory_allocator, const std::vector<OBJModelInfo>& obj_models_info)
{
    for (auto& obj_model_info : obj_models_info)
    {
        models.emplace_back(std::make_shared<OBJModel>(memory_allocator, obj_model_info));
    }
}

OBJModel::OBJModel(const std::unique_ptr<MemoryAllocator>& memory_allocator, const OBJModelInfo& obj_model_info)
        : Model(obj_model_info.name, obj_model_info.required_material)
{
    createModel(memory_allocator, obj_model_info);
}

void OBJModel::createModel(const std::unique_ptr<MemoryAllocator>& memory_allocator, const OBJModelInfo& model_info)
{
    createVertexBuffers(memory_allocator, model_info.vertices);
    createIndexBuffers(memory_allocator, model_info.indices);
}

void OBJModel::createVertexBuffers(const std::unique_ptr<MemoryAllocator>& memory_allocator, const std::vector<Vertex>& vertices)
{
    vertex_count = static_cast<uint32_t>(vertices.size());
    assert(vertex_count >= 3 && "Vertex count must be at least 3.");

    auto staging_buffer = memory_allocator->createStagingBuffer(sizeof(Vertex), vertices.size(),(void*)vertices.data());
    vertex_buffer = memory_allocator->createBuffer(
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

void OBJModel::createIndexBuffers(const std::unique_ptr<MemoryAllocator>& memory_allocator, const std::vector<uint32_t>& indices)
{
    index_count = static_cast<uint32_t>(indices.size());

    auto staging_buffer = memory_allocator->createStagingBuffer(sizeof(uint32_t), indices.size(), (void*)indices.data());
    index_buffer = memory_allocator->createBuffer(
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

void OBJModel::bind(VkCommandBuffer command_buffer)
{
    if (models.empty())
    {
        Model::bind(command_buffer);
        return;
    }

    models[0]->bind(command_buffer);
}

void OBJModel::draw(VkCommandBuffer command_buffer) const
{
    if (models.empty())
    {
        Model::draw(command_buffer);
        return;
    }

    models[0]->draw(command_buffer);
}

std::vector<std::string> OBJModel::getRequiredMaterials() const
{
    if (models.empty())
    {
        return {required_material};
    }

    std::vector<std::string> required_materials;
    required_materials.reserve(models.size());
    for (auto& model : models)
    {
        auto new_required_materials = model->getRequiredMaterials();
        required_materials.insert(required_materials.end(), new_required_materials.begin(), new_required_materials.end());
    }

    return required_materials;
}

std::vector<ModelDescription> OBJModel::getModelDescriptions() const
{
    if (models.empty())
    {
        return Model::getModelDescriptions();
    }

    std::vector<ModelDescription> model_descriptions;
    model_descriptions.reserve(models.size());
    for (auto& model : models)
    {
        auto new_model_descriptions = model->getModelDescriptions();
        model_descriptions.insert(model_descriptions.end(), new_model_descriptions.begin(), new_model_descriptions.end());
    }

    return model_descriptions;
}
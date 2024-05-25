#include "OBJModel.h"

#include <unordered_map>
#include <iostream>

#include "Utils/Algorithms.h"

OBJModel::OBJModel(VulkanFacade& device, const std::vector<OBJModelInfo>& obj_models_info, std::string name)
    : Model(std::move(name)), device{device}
{
    createManyModels(obj_models_info);
}

void OBJModel::createManyModels(const std::vector<OBJModelInfo>& obj_models_info)
{
    for (auto& obj_model_info : obj_models_info)
    {
        models.emplace_back(std::make_shared<OBJModel>(device, obj_model_info));
    }
}

OBJModel::OBJModel(VulkanFacade& device, const OBJModelInfo& obj_model_info)
        : Model(obj_model_info.name, obj_model_info.required_material), device{device}
{
    createModel(obj_model_info);
}

void OBJModel::createModel(const OBJModelInfo& model_info)
{
    createVertexBuffers(model_info.vertices);
    createIndexBuffers(model_info.indices);
}

void OBJModel::createVertexBuffers(const std::vector<Vertex>& vertices)
{
    vertex_count = static_cast<uint32_t>(vertices.size());
    assert(vertex_count >= 3 && "Vertex count must be at least 3.");
    VkDeviceSize buffer_size = sizeof(vertices[0]) * vertex_count;
    uint32_t vertex_size = sizeof(vertices[0]);

    Buffer staging_buffer
    {
            device,
            vertex_size,
            vertex_count,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    };

    staging_buffer.map();
    staging_buffer.writeToBuffer((void*)vertices.data());

    vertex_buffer = std::make_unique<Buffer>
    (
            device,
            vertex_size,
            vertex_count,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT |
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
            VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT
    );
    device.copyBuffer(staging_buffer.getBuffer(), vertex_buffer->getBuffer(), buffer_size);
}

void OBJModel::createIndexBuffers(const std::vector<uint32_t>& indices)
{
    index_count = static_cast<uint32_t>(indices.size());
    VkDeviceSize buffer_size = sizeof(indices[0]) * index_count;
    uint32_t index_size = sizeof(indices[0]);

    Buffer staging_buffer
    {
            device,
            index_size,
            index_count,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    };

    staging_buffer.map();
    staging_buffer.writeToBuffer((void*)indices.data());

    index_buffer = std::make_unique<Buffer>
    (
            device,
            index_size,
            index_count,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT |
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
            VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT
    );
    device.copyBuffer(staging_buffer.getBuffer(), index_buffer->getBuffer(), buffer_size);
}

//TODO: allocate one buffer for each vertex buffer and index buffer
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
        required_materials.append_range(model->getRequiredMaterials());
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
        model_descriptions.append_range(model->getModelDescriptions());
    }

    return model_descriptions;
}
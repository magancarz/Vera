#include "Model.h"

#include <unordered_map>
#include "RenderEngine/RenderingAPI/VulkanHelper.h"

Model::Model(std::string model_name, std::string required_material)
    : name{std::move(model_name)}, required_material{std::move(required_material)} {}

void Model::bind(VkCommandBuffer command_buffer)
{
    VkBuffer buffers[] = {vertex_buffer->getBuffer()};
    VkDeviceSize offsets[] = {0};
    vkCmdBindVertexBuffers(command_buffer, 0, 1, buffers, offsets);
    vkCmdBindIndexBuffer(command_buffer, index_buffer->getBuffer(), 0, VK_INDEX_TYPE_UINT32);
}

void Model::draw(VkCommandBuffer command_buffer) const
{
    vkCmdDrawIndexed(command_buffer, index_count, 1, 0, 0, 0);
}

std::vector<ModelDescription> Model::getModelDescriptions() const
{
    assert(vertex_buffer && index_buffer && index_count >= 3 && "Model should be valid!");

    ModelDescription model_description{};
    model_description.vertex_address = vertex_buffer->getBufferDeviceAddress();
    model_description.index_address = index_buffer->getBufferDeviceAddress();
    model_description.num_of_triangles = index_count / 3;

    return {model_description};
}

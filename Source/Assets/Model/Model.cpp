#include "Model.h"

#include "RenderEngine/RenderingAPI/VulkanHelper.h"

Model::Model(std::string model_name, std::string required_material)
    : name{std::move(model_name)}, required_material{std::move(required_material)} {}

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

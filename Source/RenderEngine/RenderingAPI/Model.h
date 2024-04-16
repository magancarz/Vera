#pragma once

#include "Device.h"
#include "Vertex.h"
#include "Buffer.h"
#include "RenderEngine/Materials/Material.h"
#include "RenderEngine/Models/AccelerationStructure.h"

class Model
{
public:
    struct Builder
    {
        std::vector<Vertex> vertices{};
        std::vector<uint32_t> indices{};

        void loadModel(const std::string& filepath);
    };

    Model(Device& device, const Model::Builder& builder);
    ~Model() = default;

    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    static std::unique_ptr<Model> createModelFromFile(Device& device, const std::string& filepath);

    void bind(VkCommandBuffer command_buffer);
    void draw(VkCommandBuffer command_buffer);

    std::unique_ptr<Buffer> vertex_buffer;
    std::unique_ptr<Buffer> index_buffer;

    AccelerationStructure blas{};

private:
    Device& device;

    void createVertexBuffers(const std::vector<Vertex>& vertices);

    uint32_t vertex_count;

    void createIndexBuffers(const std::vector<uint32_t>& indices);

    bool has_index_buffer{false};
    uint32_t index_count;

    void createBlas();
};
#pragma once

#include "Model.h"
#include "Assets/AssetManager.h"
#include "OBJModelInfo.h"
#include "RenderEngine/Memory/MemoryAllocator.h"

class OBJModel : public Model
{
public:
    OBJModel(const std::unique_ptr<MemoryAllocator>& memory_allocator, const std::vector<OBJModelInfo>& obj_models_info, std::string name);
    OBJModel(const std::unique_ptr<MemoryAllocator>& memory_allocator, const OBJModelInfo& obj_model_info);

    OBJModel(const OBJModel&) = delete;
    OBJModel& operator=(const OBJModel&) = delete;

    void bind(VkCommandBuffer command_buffer) override;
    void draw(VkCommandBuffer command_buffer) const override;

    [[nodiscard]] std::vector<std::string> getRequiredMaterials() const override;
    [[nodiscard]] std::vector<ModelDescription> getModelDescriptions() const override;

private:
    void createManyModels(const std::unique_ptr<MemoryAllocator>& memory_allocator, const std::vector<OBJModelInfo>& obj_models_info);
    void createModel(const std::unique_ptr<MemoryAllocator>& memory_allocator, const OBJModelInfo& model_info);
    void createVertexBuffers(const std::unique_ptr<MemoryAllocator>& memory_allocator, const std::vector<Vertex>& vertices);
    void createIndexBuffers(const std::unique_ptr<MemoryAllocator>& memory_allocator, const std::vector<uint32_t>& indices);

    std::vector<std::shared_ptr<Model>> models;
};

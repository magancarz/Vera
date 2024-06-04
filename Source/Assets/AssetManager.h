#pragma once

#include <memory>
#include <unordered_map>

#include "RenderEngine/RenderingAPI/VulkanFacade.h"
#include "Project/Project.h"
#include "RenderEngine/Materials/Material.h"
#include "Model/Model.h"
#include "RenderEngine/Memory/MemoryAllocator.h"
#include "Mesh.h"

class Texture;

class AssetManager
{
public:
    AssetManager(VulkanFacade& vulkan_facade, MemoryAllocator& memory_allocator);
    virtual ~AssetManager() = default;

    AssetManager(const AssetManager&) = delete;
    AssetManager operator=(const AssetManager&) = delete;

    void loadAssetsRequiredForProject(const ProjectInfo& project_info);
    void clearResources();

    virtual Mesh* fetchMesh(const std::string& mesh_name);
    Mesh* storeMesh(std::unique_ptr<Mesh> mesh);

    virtual Model* fetchModel(const std::string& model_name);
    Model* storeModel(std::unique_ptr<Model> model);

    virtual Material* fetchMaterial(const std::string& material_name);
    virtual std::vector<Material*> fetchRequiredMaterials(const std::vector<std::string>& required_materials);
    Material* storeMaterial(std::unique_ptr<Material> material);

    virtual Texture* fetchTexture(const std::string& texture_name, VkFormat image_format);
    Texture* storeTexture(std::unique_ptr<Texture> texture);

protected:
    VulkanFacade& vulkan_facade;
    MemoryAllocator& memory_allocator;

    std::unordered_map<std::string, std::unique_ptr<Mesh>> available_meshes;
    std::unordered_map<std::string, std::unique_ptr<Model>> available_models;
    std::unordered_map<std::string, std::unique_ptr<Material>> available_materials;
    std::unordered_map<std::string, std::unique_ptr<Texture>> available_textures;
};

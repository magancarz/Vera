#pragma once

#include <memory>
#include <unordered_map>

#include "RenderEngine/RenderingAPI/VulkanFacade.h"
#include "Project/Project.h"
#include "RenderEngine/Materials/Material.h"
#include "Model/Model.h"
#include "RenderEngine/Memory/MemoryAllocator.h"

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

    virtual Model* fetchModel(const std::string& model_name);
    virtual Model* storeModel(std::unique_ptr<Model> model);

    virtual Material* fetchMaterial(const std::string& material_name);
    virtual Material* storeMaterial(std::unique_ptr<Material> material);
    virtual std::vector<Material*> fetchRequiredMaterials(const std::vector<std::string>& required_materials);

    virtual Texture* fetchTexture(const std::string& texture_name, VkFormat image_format);
    virtual Texture* storeTexture(std::unique_ptr<Texture> texture);

protected:
    VulkanFacade& vulkan_facade;
    MemoryAllocator& memory_allocator;

    std::unordered_map<std::string, std::unique_ptr<Model>> available_models;
    std::unordered_map<std::string, std::unique_ptr<Material>> available_materials;
    std::unordered_map<std::string, std::unique_ptr<Texture>> available_textures;
};

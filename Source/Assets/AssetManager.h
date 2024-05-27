#pragma once

#include <memory>
#include <unordered_map>
#include <mutex>

#include "RenderEngine/RenderingAPI/VulkanFacade.h"
#include "Project/Project.h"
#include "RenderEngine/Models/Model.h"
#include "RenderEngine/Memory/MemoryAllocator.h"

class AssetManager
{
public:
    AssetManager(VulkanFacade& vulkan_facade, std::unique_ptr<MemoryAllocator>& memory_allocator);
    virtual ~AssetManager() = default;

    AssetManager(const AssetManager&) = delete;
    AssetManager operator=(const AssetManager&) = delete;

    void loadNeededAssetsForProject(const ProjectInfo& project_info);
    void clearResources();

    virtual std::shared_ptr<Model> fetchModel(const std::string& model_name);
    virtual std::shared_ptr<Material> fetchMaterial(const std::string& material_name);
    void loadMaterial(std::shared_ptr<Material> material) { available_materials.emplace(material->getName(), std::move(material)); }
    virtual std::vector<std::shared_ptr<Material>> fetchRequiredMaterials(const std::shared_ptr<Model>& model);
    virtual std::shared_ptr<Texture> fetchTexture(const std::string& texture_name, VkFormat image_format = VK_FORMAT_R8G8B8A8_SRGB);

protected:
    VulkanFacade& vulkan_facade;
    std::unique_ptr<MemoryAllocator>& memory_allocator;

    std::unordered_map<std::string, std::shared_ptr<Model>> available_models;
    std::unordered_map<std::string, std::shared_ptr<Material>> available_materials;
    std::unordered_map<std::string, std::shared_ptr<Texture>> available_textures;
};

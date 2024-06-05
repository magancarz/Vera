#pragma once

#include <memory>
#include <unordered_map>

#include "RenderEngine/RenderingAPI/VulkanFacade.h"
#include "Project/Project.h"
#include "Assets/Material/Material.h"
#include "Model/Model.h"
#include "Memory/MemoryAllocator.h"
#include "Mesh.h"

class DeviceTexture;

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

    DeviceTexture* fetchDiffuseTexture(const std::string& texture_name);
    DeviceTexture* fetchNormalMap(const std::string& texture_name);
    DeviceTexture* storeTexture(std::unique_ptr<DeviceTexture> texture);

protected:
    VulkanFacade& vulkan_facade;
    MemoryAllocator& memory_allocator;

    std::unordered_map<std::string, std::unique_ptr<Mesh>> available_meshes;
    std::unordered_map<std::string, std::unique_ptr<Model>> available_models;
    std::unordered_map<std::string, std::unique_ptr<Material>> available_materials;

    virtual DeviceTexture* fetchTexture(const std::string& texture_name, VkFormat format);

    std::unordered_map<std::string, std::unique_ptr<DeviceTexture>> available_textures;
};

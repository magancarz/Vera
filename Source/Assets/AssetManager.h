#pragma once

#include <memory>
#include <unordered_map>

#include "RenderEngine/RenderingAPI/VulkanHandler.h"
#include "Project/Project.h"
#include "Assets/Material/Material.h"
#include "Model/Model.h"
#include "Memory/MemoryAllocator.h"
#include "Mesh.h"
#include "MeshData.h"
#include "Material/MaterialData.h"

class DeviceTexture;

class AssetManager
{
public:
    AssetManager(VulkanHandler& vulkan_facade, MemoryAllocator& memory_allocator);
    virtual ~AssetManager() = default;

    AssetManager(const AssetManager&) = delete;
    AssetManager operator=(const AssetManager&) = delete;

    virtual Mesh* fetchMesh(const std::string& mesh_name);

    virtual Model* fetchModel(const std::string& model_name);

    virtual Material* fetchMaterial(const std::string& material_name);
    virtual std::vector<Material*> fetchRequiredMaterials(const std::vector<std::string>& required_materials);

    DeviceTexture* fetchDiffuseTexture(const std::string& texture_name);
    DeviceTexture* fetchNormalMap(const std::string& texture_name);

protected:
    VulkanHandler& vulkan_facade;
    MemoryAllocator& memory_allocator;

    Mesh* storeMesh(const MeshData& mesh_data);

    std::unordered_map<std::string, std::unique_ptr<Mesh>> available_meshes;

    Model* storeModel(const ModelData& model_data);

    std::unordered_map<std::string, std::unique_ptr<Model>> available_models;

    Material* storeMaterial(const MaterialData& material_data);

    std::unordered_map<std::string, std::unique_ptr<Material>> available_materials;

    virtual DeviceTexture* fetchTexture(const std::string& texture_name, VkFormat format);
    DeviceTexture* storeTexture(TextureData texture_data, VkFormat format);

    std::unordered_map<std::string, std::unique_ptr<DeviceTexture>> available_textures;
};

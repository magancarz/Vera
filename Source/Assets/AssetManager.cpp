#include "AssetManager.h"

#include "Defines.h"
#include "Logs/LogSystem.h"
#include "OBJ/OBJLoader.h"
#include "Assets/Material/VeraMaterialLoader.h"
#include "Texture/TextureLoader.h"

AssetManager::AssetManager(VulkanHandler& vulkan_facade, MemoryAllocator& memory_allocator)
    : vulkan_facade{vulkan_facade}, memory_allocator{memory_allocator} {}

Mesh* AssetManager::fetchMesh(const std::string& mesh_name)
{
    LogSystem::log(LogSeverity::LOG, "Fetching mesh named ", mesh_name.c_str());
    if (available_meshes.contains(mesh_name))
    {
        LogSystem::log(LogSeverity::LOG, "Mesh found in available meshes list. Returning...");
        return available_meshes[mesh_name].get();
    }

    LogSystem::log(LogSeverity::LOG, "Mesh was not found in available meshes list. Loading from file...");
    return storeMesh(OBJLoader::loadMeshFromFile(mesh_name));
}

Mesh* AssetManager::storeMesh(const MeshData& mesh_data)
{
    if (available_meshes.contains(mesh_data.name))
    {
        LogSystem::log(LogSeverity::WARNING, "Tried to store mesh ", mesh_data.name, " that already existed in asset manager!");
        return available_meshes.at(mesh_data.name).get();
    }

    auto mesh = std::make_unique<Mesh>();
    mesh->name = mesh_data.name;
    mesh->materials.reserve(mesh_data.materials_data.size());
    for (auto& material_data : mesh_data.materials_data)
    {
        mesh->materials.emplace_back(storeMaterial(material_data));
    }

    mesh->models.reserve(mesh_data.models_data.size());
    for (auto& model_data : mesh_data.models_data)
    {
        mesh->models.emplace_back(storeModel(model_data));
    }

    available_meshes.try_emplace(mesh_data.name, std::move(mesh));
    return available_meshes[mesh_data.name].get();
}

Material* AssetManager::storeMaterial(const MaterialData& material_data)
{
    if (available_materials.contains(material_data.name))
    {
        LogSystem::log(LogSeverity::WARNING, "Tried to store material ", material_data.name, " that already existed in asset manager!");
        return available_materials.at(material_data.name).get();
    }

    MaterialInfo material_info{};
    material_info.name = material_data.name;
    material_info.diffuse_texture = fetchDiffuseTexture(material_data.diffuse_texture_name);
    material_info.normal_texture = fetchNormalMap(material_data.normal_map_name);
    available_materials.try_emplace(material_data.name, std::make_unique<Material>(material_info));
    return available_materials[material_data.name].get();
}

Model* AssetManager::storeModel(const ModelData& model_data)
{
    if (available_models.contains(model_data.name))
    {
        LogSystem::log(LogSeverity::WARNING, "Tried to store material ", model_data.name, " that already existed in asset manager!");
        return available_models.at(model_data.name).get();
    }

    available_models.try_emplace(model_data.name, std::make_unique<Model>(memory_allocator, model_data));
    return available_models[model_data.name].get();
}

Model* AssetManager::fetchModel(const std::string& model_name)
{
    LogSystem::log(LogSeverity::LOG, "Fetching model named ", model_name.c_str());
    if (available_models.contains(model_name))
    {
        LogSystem::log(LogSeverity::LOG, "Model found in available models list. Returning...");
        return available_models[model_name].get();
    }

    LogSystem::log(LogSeverity::WARNING, "Model was not found in available models list. Searching for model in mesh file...");
    fetchMesh(model_name);
    return available_models[model_name].get();
}

Material* AssetManager::fetchMaterial(const std::string& material_name)
{
    LogSystem::log(LogSeverity::LOG, "Fetching material named ", material_name.c_str());
    if (available_materials.contains(material_name))
    {
        LogSystem::log(LogSeverity::LOG, "Material found in available materials list. Returning...");
        return available_materials[material_name].get();
    }

    LogSystem::log(LogSeverity::LOG, "Material was not found in available materials list. Loading from file...");
    return storeMaterial(VeraMaterialLoader::loadAssetFromFile(material_name));
}

std::vector<Material*> AssetManager::fetchRequiredMaterials(const std::vector<std::string>& required_materials)
{
    std::vector<Material*> materials(required_materials.size());
    for (size_t i = 0; i < required_materials.size(); ++i)
    {
        materials[i] = fetchMaterial(required_materials[i]);
    }

    return materials;
}

DeviceTexture* AssetManager::fetchDiffuseTexture(const std::string& texture_name)
{
    return fetchTexture(texture_name, Assets::DIFFUSE_TEXTURE_FORMAT);
}

DeviceTexture* AssetManager::fetchNormalMap(const std::string& texture_name)
{
    return fetchTexture(texture_name, Assets::NORMAL_MAP_FORMAT);
}

DeviceTexture* AssetManager::fetchTexture(const std::string& texture_name, VkFormat format)
{
    LogSystem::log(LogSeverity::LOG, "Fetching texture named ", texture_name.c_str());
    if (available_textures.contains(texture_name))
    {
        LogSystem::log(LogSeverity::LOG, "Texture found in available textures list. Returning...");
        return available_textures[texture_name].get();
    }

    LogSystem::log(LogSeverity::LOG, "Texture was not found in available textures list. Loading from file...");
    return storeTexture(TextureLoader::loadFromAssetFile(texture_name), format);
}

DeviceTexture* AssetManager::storeTexture(const TextureData& texture_data, VkFormat format)
{
    if (available_textures.contains(texture_data.name))
    {
        LogSystem::log(LogSeverity::WARNING, "Tried to store texture ", texture_data.name, " that already existed in asset manager!");
        return available_textures.at(texture_data.name).get();
    }

    VkImageCreateInfo image_create_info{};
    image_create_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image_create_info.imageType = VK_IMAGE_TYPE_2D;
    image_create_info.extent.width = texture_data.width;
    image_create_info.extent.height = texture_data.height;
    image_create_info.mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(texture_data.width, texture_data.height)))) + 1;
    image_create_info.format = format;
    image_create_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    image_create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    image_create_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    image_create_info.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    image_create_info.samples = VK_SAMPLE_COUNT_1_BIT;

    std::unique_ptr<Image> image = memory_allocator.createImage(image_create_info);
    available_textures.try_emplace(
        texture_data.name,
        std::make_unique<DeviceTexture>(
            vulkan_facade,
            memory_allocator,
            texture_data,
            image_create_info,
            std::move(image)));
    return available_textures.at(texture_data.name).get();
}

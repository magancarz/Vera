#include "VeraMaterial.h"
#include "MaterialDefines.h"

#include <fstream>

#include "Logs/LogSystem.h"

std::shared_ptr<VeraMaterial> VeraMaterial::fromAssetFile(const std::unique_ptr<MemoryAllocator>& memory_allocator, AssetManager* const asset_manager, const std::string& asset_name)
{
    LogSystem::log(LogSeverity::LOG, "Trying to load material named ", asset_name.c_str(), "...");

    const std::string filepath = (paths::MATERIALS_DIRECTORY_PATH / asset_name).generic_string() + ".mat";
    std::ifstream file_stream(filepath);
    std::string input;

    if (file_stream.is_open() && getline(file_stream, input))
    {
        std::stringstream iss(input);
        std::string material_name;
        iss >> material_name;

        std::string type;
        iss >> type;

        MaterialInfo material_info{};
        std::string color_x, color_y, color_z;
        iss >> color_x;
        iss >> color_y;
        iss >> color_z;

        std::string value;
        iss >> value;
        material_info.brightness = std::stoi(value);

        iss >> value;
        material_info.fuzziness = std::stof(value);

        iss >> value;
        material_info.refractive_index = std::stof(value);

        std::string texture_name;
        iss >> texture_name;

        file_stream.close();

        std::shared_ptr<Texture> diffuse_texture = asset_manager->fetchTexture(texture_name);
        std::shared_ptr<Texture> normal_texture = asset_manager->fetchTexture("blue.png");

        LogSystem::log(LogSeverity::LOG, "Loading material from file ended in success");
        return std::make_shared<VeraMaterial>(memory_allocator, material_info, std::move(material_name), std::move(diffuse_texture), std::move(normal_texture));
    }

    LogSystem::log(LogSeverity::ERROR, "Failed to load material");

    return nullptr;
}

VeraMaterial::VeraMaterial(
        const std::unique_ptr<MemoryAllocator>& memory_allocator,
        const MaterialInfo& material_info,
        std::string material_name,
        std::shared_ptr<Texture> diffuse_texture,
        std::shared_ptr<Texture> normal_texture)
    : Material(material_info, std::move(material_name), std::move(diffuse_texture), std::move(normal_texture))
{
    createMaterialInfoBuffer(memory_allocator);
}

void VeraMaterial::createMaterialInfoBuffer(const std::unique_ptr<MemoryAllocator>& memory_allocator)
{
    material_info_buffer = memory_allocator->createBuffer
    (
            sizeof(MaterialInfo),
            1,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT
    );
    auto staging_buffer = memory_allocator->createStagingBuffer(sizeof(MaterialInfo), 1, &material_info);
    material_info_buffer->copyFromBuffer(staging_buffer);
}

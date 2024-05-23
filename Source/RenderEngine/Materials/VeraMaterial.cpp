#include "VeraMaterial.h"
#include "MaterialDefines.h"

#include <fstream>

std::shared_ptr<VeraMaterial> VeraMaterial::fromAssetFile(VulkanFacade& vulkan_facade, const std::string& asset_name)
{
    printf("Trying to load material named %s...\n", asset_name.c_str());

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
        glm::vec3 color{std::stof(color_x), std::stof(color_y), std::stof(color_z)};
        material_info.color = color;

        std::string value;
        iss >> value;
        material_info.brightness = std::stoi(value);

        iss >> value;
        material_info.fuzziness = std::stof(value);

        iss >> value;
        material_info.refractive_index = std::stof(value);

        file_stream.close();

        printf("Loading material from file ended in success\n");
        return std::make_shared<VeraMaterial>(vulkan_facade, material_info, std::move(material_name));
    }

    printf("Failed to load material\n");

    return nullptr;
}

VeraMaterial::VeraMaterial(
        VulkanFacade& vulkan_facade,
        const MaterialInfo& material_info,
        std::string material_name)
    : Material(material_info, std::move(material_name))
{
    createMaterialInfoBuffer(vulkan_facade);
    assignMaterialHitGroupIndex();
}

void VeraMaterial::createMaterialInfoBuffer(VulkanFacade& vulkan_facade)
{
    material_info_buffer = std::make_unique<Buffer>
    (
            vulkan_facade,
            sizeof(MaterialInfo),
            1,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT
    );
    material_info_buffer->writeWithStagingBuffer(&this->material_info);
}

void VeraMaterial::assignMaterialHitGroupIndex()
{
    if (material_info.brightness > 0)
    {
        material_hit_group_index = defines::material_indices::light_hit_group_index;
    }
    else if (material_info.fuzziness > 0)
    {
        material_hit_group_index = defines::material_indices::specular_hit_group_index;
    }
}
#include "MaterialBuilder.h"

#include <fstream>

#include "MaterialDefines.h"

MaterialBuilder::MaterialBuilder(VulkanFacade& device)
    : device{device} {}

MaterialBuilder& MaterialBuilder::fromAssetFile(const std::string& asset_name)
{
    printf("Trying to load material named %s...\n", asset_name.c_str());

    const std::string filepath = (paths::MATERIALS_DIRECTORY_PATH / asset_name).generic_string() + ".mat";
    std::ifstream file_stream(filepath);
    std::string input;

    if (file_stream.is_open() && getline(file_stream, input))
    {
        std::stringstream iss(input);
        iss >> material_name;

        std::string type;
        iss >> type;

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
    }

    printf("Failed to load material\n");

    return *this;
}

MaterialBuilder& MaterialBuilder::lambertian()
{
    material_hit_group_index = defines::material_indices::lambertian_hit_group_index;

    return *this;
}

MaterialBuilder& MaterialBuilder::color(const glm::vec3& color)
{
    material_info.color = color;

    return *this;
}

MaterialBuilder& MaterialBuilder::specular()
{
    material_hit_group_index = defines::material_indices::specular_hit_group_index;

    return *this;
}

MaterialBuilder& MaterialBuilder::fuzziness(float value)
{
    assert(material_info.brightness < 0 && material_info.refractive_index < 0 && "Material shouldn't have values defined for other types of materials");
    assert(material_hit_group_index == defines::material_indices::specular_hit_group_index && "Material should be specular");

    material_info.fuzziness = value;

    return *this;
}

MaterialBuilder& MaterialBuilder::light()
{
    material_hit_group_index = defines::material_indices::light_hit_group_index;

    return *this;
}

MaterialBuilder& MaterialBuilder::brightness(unsigned int value)
{
    assert(material_info.fuzziness < 0 && material_info.refractive_index < 0 && "Material shouldn't have values defined for other types of materials");
    assert(material_hit_group_index == defines::material_indices::light_hit_group_index && "Material should be specular");

    material_info.brightness = value;

    return *this;
}

MaterialBuilder& MaterialBuilder::name(std::string name)
{
    material_name = std::move(name);

    return *this;
}

std::shared_ptr<Material> MaterialBuilder::build()
{
    auto material = std::make_shared<Material>(material_info, std::move(material_name));
    material->material_info_buffer = std::make_unique<Buffer>
    (
            device,
            sizeof(MaterialInfo),
            1,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT
    );
    material->material_info_buffer->writeWithStagingBuffer(&material_info);
    material->material_hit_group_index = material_hit_group_index;
    return material;
}
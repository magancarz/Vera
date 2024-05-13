#include <iostream>
#include <fstream>
#include "Material.h"
#include "Utils/VeraDefines.h"

std::shared_ptr<Material> Material::loadMaterialFromFile(Device& device, const std::string& material_name)
{
    printf("Trying to load material named %s...\n", material_name.c_str());

    const std::string filepath = (paths::MATERIALS_DIRECTORY_PATH / material_name).generic_string() + ".mat";
    std::ifstream file_stream(filepath);
    std::string input;

    MaterialInfo material_info{};
    if (file_stream.is_open() && getline(file_stream, input))
    {
        std::stringstream iss(input);
        std::string name;
        iss >> name;

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
        return std::make_shared<Material>(device, material_info);
    }

    printf("Failed to load material\n");
    return nullptr;
}

Material::Material(Device& device, MaterialInfo in_material_info)
    : device{device}, material_info{in_material_info}
{
    createMaterialBuffer();
}

void Material::assignMaterialHitGroup(BlasInstance& blas_instance) const
{
    //TODO: change it ofc
    uint32_t shader_off = 0;
    if (material_info.brightness > 0)
    {
        shader_off = 1;
    }
    else if (material_info.fuzziness >= 0)
    {
        shader_off = 2;
    }

    blas_instance.bottomLevelAccelerationStructureInstance.instanceShaderBindingTableRecordOffset = shader_off;
}

void Material::createMaterialBuffer()
{
    material_info_buffer = std::make_unique<Buffer>
    (
        device,
        sizeof(MaterialInfo),
        1,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT
    );
    material_info_buffer->writeWithStagingBuffer(&material_info);
}

void Material::getMaterialDescription(ObjectDescription& object_description)
{
    object_description.material_address = material_info_buffer->getBufferDeviceAddress();
}
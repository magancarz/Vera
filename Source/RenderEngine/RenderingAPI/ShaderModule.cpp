#include "ShaderModule.h"
#include "Utils/VeraDefines.h"
#include "VulkanDefines.h"

#include <fstream>

ShaderModule::ShaderModule(VulkanFacade& device, const std::string& shader_code_file, VkShaderStageFlagBits shader_stage)
    : device{device}
{
    std::string path_to_shader_code = getPathToShaderCodeFile(shader_code_file, shader_stage);
    createShaderModule(path_to_shader_code);
}

std::string ShaderModule::getPathToShaderCodeFile(const std::string& shader_code_file, VkShaderStageFlagBits shader_stage)
{
    if (!SHADER_CODE_EXTENSIONS.contains(shader_stage))
    {
        throw std::runtime_error("Unsupported shader stage extension while creating shader module!");
    }
    std::filesystem::path file_name_with_extension{shader_code_file + SHADER_CODE_EXTENSIONS.at(shader_stage)};
    std::filesystem::path full_file_path{paths::SHADERS_DIRECTORY_PATH / file_name_with_extension};
    return full_file_path.generic_string();
}

void ShaderModule::createShaderModule(const std::string& path_to_shader_code)
{
    std::vector<uint32_t> shader_source = loadShaderSourceCode(path_to_shader_code);

    VkShaderModuleCreateInfo shader_module_create_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    shader_module_create_info.codeSize = static_cast<uint32_t>(shader_source.size() * sizeof(uint32_t));
    shader_module_create_info.pCode = shader_source.data();

    if (vkCreateShaderModule(device.getDevice(), &shader_module_create_info, VulkanDefines::NO_CALLBACK, &shader_module) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create shader module!");
    }
}

std::vector<uint32_t> ShaderModule::loadShaderSourceCode(const std::string& path_to_shader_code)
{
    std::ifstream file(path_to_shader_code, std::ios::binary | std::ios::ate);
    std::streamsize file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<uint32_t> shader_source(file_size / sizeof(uint32_t));

    file.read(reinterpret_cast<char*>(shader_source.data()), file_size);
    file.close();

    return shader_source;
}

ShaderModule::~ShaderModule()
{
    vkDestroyShaderModule(device.getDevice(), shader_module, VulkanDefines::NO_CALLBACK);
}


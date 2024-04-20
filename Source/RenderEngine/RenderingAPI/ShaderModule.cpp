#include "ShaderModule.h"
#include "VulkanDefines.h"

#include <fstream>

ShaderModule::ShaderModule(Device& device, const std::string& path_to_shader_code)
    : device{device}
{
    createShaderModule(path_to_shader_code);
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


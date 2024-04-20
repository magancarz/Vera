#pragma once

#include <string>

#include <vulkan/vulkan.hpp>

#include "Device.h"

class ShaderModule
{
public:
    ShaderModule(Device& device, const std::string& path_to_shader_code);
    ~ShaderModule();

    ShaderModule(const ShaderModule&) = delete;
    ShaderModule& operator=(const ShaderModule&) = delete;

    VkShaderModule getShaderModule() const { return shader_module; }

private:
    Device& device;

    void createShaderModule(const std::string& path_to_shader_code);
    std::vector<uint32_t> loadShaderSourceCode(const std::string& path_to_shader_code);

    VkShaderModule shader_module{VK_NULL_HANDLE};
};

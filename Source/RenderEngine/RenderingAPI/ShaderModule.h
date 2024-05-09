#pragma once

#include <string>
#include <filesystem>
#include <unordered_map>

#include <vulkan/vulkan.hpp>

#include "Device.h"

class ShaderModule
{
public:
    ShaderModule(Device& device, const std::string& shader_code_file, VkShaderStageFlagBits shader_stage);
    ~ShaderModule();

    ShaderModule(const ShaderModule&) = delete;
    ShaderModule& operator=(const ShaderModule&) = delete;

    [[nodiscard]] VkShaderModule getShaderModule() const { return shader_module; }

    inline static const std::unordered_map<VkShaderStageFlagBits, const char* const> SHADER_CODE_EXTENSIONS
    {
            {VK_SHADER_STAGE_RAYGEN_BIT_KHR, ".rgen.spv"},
            {VK_SHADER_STAGE_MISS_BIT_KHR, ".rmiss.spv"},
            {VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, ".rchit.spv"},
            {VK_SHADER_STAGE_ANY_HIT_BIT_KHR, ".rahit.spv"}
    };

private:
    Device& device;

    static std::string getPathToShaderCodeFile(const std::string& shader_code_file, VkShaderStageFlagBits shader_stage);
    void createShaderModule(const std::string& path_to_shader_code);
    static std::vector<uint32_t> loadShaderSourceCode(const std::string& path_to_shader_code);

    VkShaderModule shader_module{VK_NULL_HANDLE};
};

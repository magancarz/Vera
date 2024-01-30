#pragma once

#include <string>
#include <vector>

#include "PipelineConfigInfo.h"
#include "Device.h"

class Pipeline
{
public:
    Pipeline(
        Device& device,
        const std::string& vertex_file_path,
        const std::string& fragment_file_path,
        const PipelineConfigInfo& config_info);
    ~Pipeline();

    Pipeline(const Pipeline&) = delete;
    void operator=(const Pipeline&) = delete;

    void bind(VkCommandBuffer command_buffer);
    static PipelineConfigInfo defaultPipelineConfigInfo(uint32_t width, uint32_t height);

private:
    static std::vector<char> readFile(const std::string& file_path);
    void createGraphicsPipeline(const std::string& vertex_file_path, const std::string& fragment_file_path, const PipelineConfigInfo& config_info);
    void createShaderModule(const std::vector<char>& code, VkShaderModule* shader_module);

    Device& device;
    VkPipeline graphics_pipeline;
    VkShaderModule vertex_shader_module;
    VkShaderModule fragment_shader_module;
};

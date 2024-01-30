#pragma once

#include <vulkan/vulkan.hpp>

struct PipelineConfigInfo
{
    VkViewport viewport;
    VkRect2D scissor;
    VkPipelineInputAssemblyStateCreateInfo input_assembly_info{};
    VkPipelineRasterizationStateCreateInfo rasterization_info{};
    VkPipelineMultisampleStateCreateInfo multisample_info{};
    VkPipelineColorBlendAttachmentState color_blend_attachment{};
    VkPipelineColorBlendStateCreateInfo color_blend_info{};
    VkPipelineDepthStencilStateCreateInfo depth_stencil_info{};
    VkPipelineLayout pipeline_layout = nullptr;
    VkRenderPass render_pass = nullptr;
    uint32_t subpass = 0;
};
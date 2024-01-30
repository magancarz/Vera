#include "Pipeline.h"

#include <fstream>

#include "VulkanDefines.h"

Pipeline::Pipeline(
        Device& device,
        const std::string& vertex_file_path,
        const std::string& fragment_file_path,
        const PipelineConfigInfo& config_info)
    : device{device}
{
    createGraphicsPipeline(vertex_file_path, fragment_file_path, config_info);
}

Pipeline::~Pipeline()
{
    vkDestroyShaderModule(device.getDevice(), vertex_shader_module, VulkanDefines::NO_CALLBACK);
    vkDestroyShaderModule(device.getDevice(), fragment_shader_module, VulkanDefines::NO_CALLBACK);
    vkDestroyPipeline(device.getDevice(), graphics_pipeline, VulkanDefines::NO_CALLBACK);
}

void Pipeline::bind(VkCommandBuffer command_buffer)
{
    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_pipeline);
}

void Pipeline::createGraphicsPipeline(const std::string& vertex_file_path, const std::string& fragment_file_path, const PipelineConfigInfo& config_info)
{
    assert(config_info.pipeline_layout != VK_NULL_HANDLE && "Cannot create graphics pipeline: no pipeline layout provided in config info!");
    assert(config_info.render_pass != VK_NULL_HANDLE && "Cannot create graphics pipeline: no render pass provided in config info!");

    auto vert_code = readFile(vertex_file_path);
    auto frag_code = readFile(fragment_file_path);

    createShaderModule(vert_code, &vertex_shader_module);
    createShaderModule(frag_code, &fragment_shader_module);

    VkPipelineShaderStageCreateInfo shader_stage_create_infos[2];
    shader_stage_create_infos[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shader_stage_create_infos[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    shader_stage_create_infos[0].module = vertex_shader_module;
    shader_stage_create_infos[0].pName = "main";
    shader_stage_create_infos[0].flags = 0;
    shader_stage_create_infos[0].pNext = nullptr;
    shader_stage_create_infos[0].pSpecializationInfo = nullptr;

    shader_stage_create_infos[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shader_stage_create_infos[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    shader_stage_create_infos[1].module = fragment_shader_module;
    shader_stage_create_infos[1].pName = "main";
    shader_stage_create_infos[1].flags = 0;
    shader_stage_create_infos[1].pNext = nullptr;
    shader_stage_create_infos[1].pSpecializationInfo = nullptr;

    VkPipelineVertexInputStateCreateInfo vertex_input_state_create_info{};
    vertex_input_state_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertex_input_state_create_info.vertexAttributeDescriptionCount = 0;
    vertex_input_state_create_info.vertexBindingDescriptionCount = 0;
    vertex_input_state_create_info.pVertexAttributeDescriptions = nullptr;
    vertex_input_state_create_info.pVertexBindingDescriptions = nullptr;

    VkPipelineViewportStateCreateInfo viewport_info{};
    viewport_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewport_info.viewportCount = 1;
    viewport_info.pViewports = &config_info.viewport;
    viewport_info.scissorCount = 1;
    viewport_info.pScissors = &config_info.scissor;

    VkGraphicsPipelineCreateInfo pipeline_create_info{};
    pipeline_create_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipeline_create_info.stageCount = 2;
    pipeline_create_info.pStages = shader_stage_create_infos;
    pipeline_create_info.pVertexInputState = &vertex_input_state_create_info;
    pipeline_create_info.pInputAssemblyState = &config_info.input_assembly_info;
    pipeline_create_info.pViewportState = &viewport_info;
    pipeline_create_info.pRasterizationState = &config_info.rasterization_info;
    pipeline_create_info.pMultisampleState = &config_info.multisample_info;
    pipeline_create_info.pColorBlendState = &config_info.color_blend_info;
    pipeline_create_info.pDepthStencilState = &config_info.depth_stencil_info;
    pipeline_create_info.pDynamicState = nullptr;

    pipeline_create_info.layout = config_info.pipeline_layout;
    pipeline_create_info.renderPass = config_info.render_pass;
    pipeline_create_info.subpass = config_info.subpass;

    pipeline_create_info.basePipelineIndex = -1;
    pipeline_create_info.basePipelineHandle = VK_NULL_HANDLE;

    if (vkCreateGraphicsPipelines(device.getDevice(), VK_NULL_HANDLE, 1, &pipeline_create_info, nullptr, &graphics_pipeline) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create graphics pipeline!");
    }
}

std::vector<char> Pipeline::readFile(const std::string& file_path)
{
    std::ifstream file{file_path, std::ios::ate | std::ios::binary};
    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open file " + file_path);
    }

    auto file_size = static_cast<size_t>(file.tellg());
    std::vector<char> buffer(file_size);
    file.seekg(0);
    file.read(buffer.data(), file_size);

    file.close();
    return buffer;
}

void Pipeline::createShaderModule(const std::vector<char>& code, VkShaderModule* shader_module)
{
    VkShaderModuleCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    create_info.codeSize = code.size();
    create_info.pCode = reinterpret_cast<const uint32_t*>(code.data());

    if (vkCreateShaderModule(device.getDevice(), &create_info, VulkanDefines::NO_CALLBACK, shader_module) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create shader module!");
    }
}

PipelineConfigInfo Pipeline::defaultPipelineConfigInfo(uint32_t width, uint32_t height)
{
    PipelineConfigInfo config_info{};

    config_info.input_assembly_info.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    config_info.input_assembly_info.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    config_info.input_assembly_info.primitiveRestartEnable = VK_FALSE;

    config_info.viewport.x = 0.0f;
    config_info.viewport.y = 0.0f;
    config_info.viewport.width = static_cast<float>(width);
    config_info.viewport.height = static_cast<float>(height);
    config_info.viewport.minDepth = 0.0f;
    config_info.viewport.maxDepth = 1.0f;

    config_info.scissor.offset = {0, 0};
    config_info.scissor.extent = {width, height};

    config_info.rasterization_info.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    config_info.rasterization_info.depthClampEnable = VK_FALSE;
    config_info.rasterization_info.rasterizerDiscardEnable = VK_FALSE;
    config_info.rasterization_info.polygonMode = VK_POLYGON_MODE_FILL;
    config_info.rasterization_info.lineWidth = 1.0f;
    config_info.rasterization_info.cullMode = VK_CULL_MODE_NONE;
    config_info.rasterization_info.frontFace = VK_FRONT_FACE_CLOCKWISE;
    config_info.rasterization_info.depthBiasEnable = VK_FALSE;
    config_info.rasterization_info.depthBiasConstantFactor = 0.0f;
    config_info.rasterization_info.depthBiasClamp = 0.0f;
    config_info.rasterization_info.depthBiasSlopeFactor = 0.0f;

    config_info.multisample_info.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    config_info.multisample_info.sampleShadingEnable = VK_FALSE;
    config_info.multisample_info.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    config_info.multisample_info.minSampleShading = 1.0f;
    config_info.multisample_info.pSampleMask = nullptr;
    config_info.multisample_info.alphaToCoverageEnable = VK_FALSE;
    config_info.multisample_info.alphaToOneEnable = VK_FALSE;

    config_info.color_blend_attachment.colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
            VK_COLOR_COMPONENT_A_BIT;
    config_info.color_blend_attachment.blendEnable = VK_FALSE;
    config_info.color_blend_attachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
    config_info.color_blend_attachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
    config_info.color_blend_attachment.colorBlendOp = VK_BLEND_OP_ADD;
    config_info.color_blend_attachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    config_info.color_blend_attachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    config_info.color_blend_attachment.alphaBlendOp = VK_BLEND_OP_ADD;

    config_info.color_blend_info.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    config_info.color_blend_info.logicOpEnable = VK_FALSE;
    config_info.color_blend_info.logicOp = VK_LOGIC_OP_COPY;
    config_info.color_blend_info.attachmentCount = 1;
    config_info.color_blend_info.pAttachments = &config_info.color_blend_attachment;
    config_info.color_blend_info.blendConstants[0] = 0.0f;
    config_info.color_blend_info.blendConstants[1] = 0.0f;
    config_info.color_blend_info.blendConstants[2] = 0.0f;
    config_info.color_blend_info.blendConstants[3] = 0.0f;

    config_info.depth_stencil_info.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    config_info.depth_stencil_info.depthTestEnable = VK_TRUE;
    config_info.depth_stencil_info.depthWriteEnable = VK_TRUE;
    config_info.depth_stencil_info.depthCompareOp = VK_COMPARE_OP_LESS;
    config_info.depth_stencil_info.depthBoundsTestEnable = VK_FALSE;
    config_info.depth_stencil_info.minDepthBounds = 0.0f;
    config_info.depth_stencil_info.maxDepthBounds = 1.0f;
    config_info.depth_stencil_info.stencilTestEnable = VK_FALSE;
    config_info.depth_stencil_info.front = {};
    config_info.depth_stencil_info.back = {};

    return config_info;
}

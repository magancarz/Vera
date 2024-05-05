#include "PostProcessing.h"

#include "RenderEngine/RenderingAPI/VulkanDefines.h"
#include "Objects/Components/TransformComponent.h"

PostProcessing::PostProcessing(
        Device& device,
        VkRenderPass render_pass,
        VkDescriptorSetLayout input_texture)
        : device{device}
{
    loadSceneQuad();
    createPipelineLayout(input_texture);
    createPipeline(render_pass);
}

void PostProcessing::loadSceneQuad()
{
    scene_quad = Model::createModelFromFile(device, "Resources/Models/scene_quad.obj");
}

void PostProcessing::createPipelineLayout(VkDescriptorSetLayout input_texture)
{
    std::vector<VkDescriptorSetLayout> descriptor_set_layouts{input_texture};

    VkPipelineLayoutCreateInfo pipeline_layout_info{};
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = static_cast<uint32_t>(descriptor_set_layouts.size());
    pipeline_layout_info.pSetLayouts = descriptor_set_layouts.data();
    pipeline_layout_info.pushConstantRangeCount = 0;
    pipeline_layout_info.pPushConstantRanges = nullptr;

    if (vkCreatePipelineLayout(device.getDevice(), &pipeline_layout_info, VulkanDefines::NO_CALLBACK, &pipeline_layout) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create pipeline layout!");
    }
}

void PostProcessing::createPipeline(VkRenderPass render_pass)
{
    assert(pipeline_layout != nullptr && "Cannot create pipeline before pipeline layout");

    PipelineConfigInfo config_info{};
    Pipeline::defaultPipelineConfigInfo(config_info);
    config_info.render_pass = render_pass;
    config_info.pipeline_layout = pipeline_layout;
    simple_pipeline = std::make_unique<Pipeline>
    (
            device,
            "Shaders/SimpleShader.vert.spv",
            "Shaders/SimpleShader.frag.spv",
            config_info
    );
}

PostProcessing::~PostProcessing()
{
    vkDestroyPipelineLayout(device.getDevice(), pipeline_layout, VulkanDefines::NO_CALLBACK);
}

void PostProcessing::apply(FrameInfo& frame_info)
{
    simple_pipeline->bind(frame_info.command_buffer);
    vkCmdBindDescriptorSets(
            frame_info.command_buffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            pipeline_layout,
            0,
            1,
            &frame_info.ray_traced_texture,
            0,
            nullptr);

    scene_quad->bind(frame_info.command_buffer);
    scene_quad->draw(frame_info.command_buffer);
}
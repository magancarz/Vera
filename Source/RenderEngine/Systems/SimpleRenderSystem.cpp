#include "SimpleRenderSystem.h"

#include "RenderEngine/RenderingAPI/VulkanDefines.h"
#include "Objects/Components/TransformComponent.h"

SimpleRenderSystem::SimpleRenderSystem(
        Device& device,
        VkRenderPass render_pass,
        VkDescriptorSetLayout global_uniform_buffer_set_layout,
        VkDescriptorSetLayout global_texture_set_layout)
    : device{device}
{
    createPipelineLayout(global_uniform_buffer_set_layout, global_texture_set_layout);
    createPipeline(render_pass);
}

SimpleRenderSystem::~SimpleRenderSystem()
{
    vkDestroyPipelineLayout(device.getDevice(), pipeline_layout, VulkanDefines::NO_CALLBACK);
}

void SimpleRenderSystem::createPipelineLayout(VkDescriptorSetLayout global_set_layout, VkDescriptorSetLayout global_texture_set_layout)
{
    VkPushConstantRange push_constant_range{};
    push_constant_range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    push_constant_range.offset = 0;
    push_constant_range.size = sizeof(SimplePushConstantData);

    std::vector<VkDescriptorSetLayout> descriptor_set_layouts{global_set_layout, global_texture_set_layout};

    VkPipelineLayoutCreateInfo pipeline_layout_info{};
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = static_cast<uint32_t>(descriptor_set_layouts.size());
    pipeline_layout_info.pSetLayouts = descriptor_set_layouts.data();
    pipeline_layout_info.pushConstantRangeCount = 1;
    pipeline_layout_info.pPushConstantRanges = &push_constant_range;

    if (vkCreatePipelineLayout(device.getDevice(), &pipeline_layout_info, nullptr, &pipeline_layout) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create pipeline layout!");
    }
}

void SimpleRenderSystem::createPipeline(VkRenderPass render_pass)
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

void SimpleRenderSystem::renderObjects(FrameInfo& frame_info)
{
    simple_pipeline->bind(frame_info.command_buffer);
    vkCmdBindDescriptorSets(
            frame_info.command_buffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            pipeline_layout,
            0,
            1,
            &frame_info.global_uniform_buffer_descriptor_set,
            0,
            nullptr);

    for (auto& [id, obj] : frame_info.objects)
    {
        if (obj.model)
        {
            SimplePushConstantData push{};
            auto model_matrix = obj.transform_component.transform();
            push.model = model_matrix;
            push.normal_matrix = obj.transform_component.normalMatrix();

            vkCmdBindDescriptorSets(
                    frame_info.command_buffer,
                    VK_PIPELINE_BIND_POINT_GRAPHICS,
                    pipeline_layout,
                    1,
                    1,
                    &obj.material->getDescriptorSet(frame_info.frame_index),
                    0,
                    nullptr);

            vkCmdPushConstants(
                    frame_info.command_buffer,
                    pipeline_layout,
                    VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                    0,
                    sizeof(SimplePushConstantData),
                    &push);

            obj.model->bind(frame_info.command_buffer);
            obj.model->draw(frame_info.command_buffer);
        }
    }
}

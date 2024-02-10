#include "SimpleRenderSystem.h"

#include "RenderEngine/RenderingAPI/VulkanDefines.h"

SimpleRenderSystem::SimpleRenderSystem(Device& device, Window& window, VkRenderPass render_pass)
    : device{device}, window{window}
{
    createPipelineLayout();
    createPipeline(render_pass);
}

SimpleRenderSystem::~SimpleRenderSystem()
{
    vkDestroyPipelineLayout(device.getDevice(), pipeline_layout, VulkanDefines::NO_CALLBACK);
}

void SimpleRenderSystem::createPipelineLayout()
{
    VkPushConstantRange push_constant_range{};
    push_constant_range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    push_constant_range.offset = 0;
    push_constant_range.size = sizeof(SimplePushConstantData);

    VkPipelineLayoutCreateInfo pipeline_layout_info{};
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = 0;
    pipeline_layout_info.pSetLayouts = nullptr;
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

void SimpleRenderSystem::renderObjects(
        VkCommandBuffer command_buffer,
        std::vector<Object>& objects,
        const Camera& camera)
{
    simple_pipeline->bind(command_buffer);

    for (auto& obj : objects)
    {
        SimplePushConstantData push{};
        push.transform = camera.getPerspectiveProjectionMatrix(window.getAspect()) * camera.getViewMatrix() * obj.transform_component.transform();
        push.color = obj.color;

        vkCmdPushConstants(
                command_buffer,
                pipeline_layout,
                VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                0,
                sizeof(SimplePushConstantData),
                &push
        );
        obj.model->bind(command_buffer);
        obj.model->draw(command_buffer);
    }
}

#include "PointLightSystem.h"

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

#include <array>
#include <cassert>
#include <stdexcept>

PointLightSystem::PointLightSystem(
        Device& device, VkRenderPass render_pass, VkDescriptorSetLayout global_set_layout)
        : device{device}
{
    createPipelineLayout(global_set_layout);
    createPipeline(render_pass);
}

PointLightSystem::~PointLightSystem()
{
    vkDestroyPipelineLayout(device.getDevice(), pipeline_layout, nullptr);
}

void PointLightSystem::createPipelineLayout(VkDescriptorSetLayout global_set_layout)
{
     VkPushConstantRange push_constant_range{};
     push_constant_range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
     push_constant_range.offset = 0;
     push_constant_range.size = sizeof(PointLightPushConstants);

    std::vector<VkDescriptorSetLayout> descriptor_set_layouts{global_set_layout};

    VkPipelineLayoutCreateInfo pipeline_layout_info{};
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = static_cast<uint32_t>(descriptor_set_layouts.size());
    pipeline_layout_info.pSetLayouts = descriptor_set_layouts.data();
    pipeline_layout_info.pushConstantRangeCount = 1;
    pipeline_layout_info.pPushConstantRanges = &push_constant_range;
    if (vkCreatePipelineLayout(device.getDevice(), &pipeline_layout_info, nullptr, &pipeline_layout) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create pipeline layout!");
    }
}

void PointLightSystem::createPipeline(VkRenderPass render_pass)
{
    assert(pipeline_layout != nullptr && "Cannot create pipeline before pipeline layout");

    PipelineConfigInfo pipeline_config{};
    Pipeline::defaultPipelineConfigInfo(pipeline_config);
    pipeline_config.attribute_descriptions.clear();
    pipeline_config.binding_descriptions.clear();
    pipeline_config.render_pass = render_pass;
    pipeline_config.pipeline_layout = pipeline_layout;
    pipeline = std::make_unique<Pipeline>(
            device,
            "Shaders/PointLight.vert.spv",
            "Shaders/PointLight.frag.spv",
            pipeline_config);
}

void PointLightSystem::update(FrameInfo& frame_info, GlobalUBO& ubo)
{
    auto rotate_light = glm::rotate(glm::mat4{1.f}, 0.5f * frame_info.frame_time, {0.f, -1.f, 0.f});
    int light_index = 0;
    for (auto& [id, object] : frame_info.objects)
    {
        if (object.point_light)
        {
            assert(light_index < RendererDefines::MAX_NUMBER_OF_LIGHTS && "Point lights exceed maximum specified");

            object.transform_component.translation = glm::vec3(rotate_light * glm::vec4(object.transform_component.translation, 1.f));

            ubo.point_lights[light_index].position = glm::vec4(object.transform_component.translation, 1.f);
            ubo.point_lights[light_index].color = glm::vec4(object.color, object.point_light->light_intensity);
            ++light_index;
        }
    }
    ubo.number_of_lights = light_index;
}

void PointLightSystem::render(FrameInfo& frame_info)
{
    pipeline->bind(frame_info.command_buffer);

    vkCmdBindDescriptorSets(
            frame_info.command_buffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            pipeline_layout,
            0,
            1,
            &frame_info.global_descriptor_sets[0],
            0,
            nullptr);

    for (auto& [id, object] : frame_info.objects)
    {
        if (object.point_light)
        {
            PointLightPushConstants push{};
            push.position = glm::vec4{object.transform_component.translation, 1.f};
            push.color = glm::vec4{object.color, object.point_light->light_intensity};
            push.radius = object.transform_component.scale.x;

            vkCmdPushConstants(
                    frame_info.command_buffer,
                    pipeline_layout,
                    VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                    0,
                    sizeof(PointLightPushConstants),
                    &push);
            vkCmdDraw(frame_info.command_buffer, 6, 1, 0, 0);
        }
    }

}
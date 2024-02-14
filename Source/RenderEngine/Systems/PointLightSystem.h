#pragma once

#include "RenderEngine/Camera.h"
#include "RenderEngine/RenderingAPI/Device.h"
#include "RenderEngine/FrameInfo.h"
#include "Objects/Object.h"
#include "RenderEngine/RenderingAPI/Pipeline.h"

#include <memory>
#include <vector>

class PointLightSystem {
public:
    PointLightSystem(
            Device& device, VkRenderPass render_pass, VkDescriptorSetLayout global_set_layout);
    ~PointLightSystem();

    PointLightSystem(const PointLightSystem&) = delete;
    PointLightSystem& operator=(const PointLightSystem&) = delete;

    void render(FrameInfo& frame_info);

private:
    void createPipelineLayout(VkDescriptorSetLayout global_set_layout);
    void createPipeline(VkRenderPass render_pass);

    Device& device;

    std::unique_ptr<Pipeline> pipeline;
    VkPipelineLayout pipeline_layout;
};
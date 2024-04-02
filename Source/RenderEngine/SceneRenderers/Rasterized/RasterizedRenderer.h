#pragma once

#include "RenderEngine/SceneRenderers/SceneRenderer.h"

class RasterizedRenderer : public SceneRenderer
{
public:
    RasterizedRenderer(Device& device, VkRenderPass render_pass);

    void renderScene(FrameInfo& frame_info) override;

private:
    Device& device;

    void createDescriptorPool();

    std::unique_ptr<DescriptorPool> global_pool{};
    std::vector<std::unique_ptr<Buffer>> ubo_buffers;
    std::vector<VkDescriptorSet> global_uniform_buffer_descriptor_sets;

    std::unique_ptr<SimpleRenderSystem> simple_render_system;
    std::unique_ptr<PointLightSystem> point_light_render_system;

    std::vector<std::shared_ptr<Material>> materials;
};

#pragma once

#include "RenderEngine/RenderingAPI/VulkanFacade.h"
#include "RenderEngine/FrameInfo.h"
#include "RenderEngine/RenderingAPI/Pipeline.h"
#include "Assets/AssetManager.h"

class PostProcessing
{
public:
    PostProcessing(
            VulkanFacade& device,
            AssetManager& asset_manager,
            VkRenderPass render_pass,
            VkDescriptorSetLayout input_texture);
    ~PostProcessing();

    PostProcessing(const PostProcessing&) = delete;
    PostProcessing& operator=(const PostProcessing&) = delete;

    void apply(FrameInfo& frame_info);

private:
    void loadSceneQuad(AssetManager& asset_manager);
    void createPipelineLayout(VkDescriptorSetLayout input_texture);
    void createPipeline(VkRenderPass render_pass);

    VulkanFacade& device;

    std::unique_ptr<Pipeline> simple_pipeline;
    VkPipelineLayout pipeline_layout;

    Model* scene_quad;
};
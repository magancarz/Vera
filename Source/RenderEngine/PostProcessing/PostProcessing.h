#pragma once

#include "RenderEngine/RenderingAPI/VulkanHandler.h"
#include "RenderEngine/FrameInfo.h"
#include "RenderEngine/RenderingAPI/Pipeline.h"
#include "Assets/AssetManager.h"

class PostProcessing
{
public:
    PostProcessing(
        VulkanHandler& device,
        AssetManager& asset_manager,
        VkRenderPass render_pass,
        VkDescriptorSetLayout input_texture);
    ~PostProcessing();

    PostProcessing(const PostProcessing&) = delete;
    PostProcessing& operator=(const PostProcessing&) = delete;
    PostProcessing(PostProcessing&&) = delete;
    PostProcessing& operator=(PostProcessing&&) = delete;

    void apply(FrameInfo& frame_info);

private:
    void loadSceneQuad(AssetManager& asset_manager);
    void createPipelineLayout(VkDescriptorSetLayout input_texture);
    void createPipeline(VkRenderPass render_pass);

    VulkanHandler& device;

    std::unique_ptr<Pipeline> simple_pipeline;
    VkPipelineLayout pipeline_layout;

    Model* scene_quad;
    ModelDescription scene_quad_model_description;
};
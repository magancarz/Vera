#include "PostProcessingChainRenderer.h"

#include "RenderEngine/PostProcessing/Bloom/BloomEffectRenderer.h"

PostProcessingChainRenderer::PostProcessingChainRenderer()
{
    post_processing_renderers.emplace_back(std::make_unique<BloomEffectRenderer>());
}

void PostProcessingChainRenderer::applyPostProcessing(const utils::Texture& in_out_hdr_color_buffer)
{
    for (const auto& post_processing_renderer : post_processing_renderers)
    {
        post_processing_renderer->apply(in_out_hdr_color_buffer);
    }
}
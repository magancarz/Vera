#pragma once

#include "PostProcessingRenderer.h"

class PostProcessingChainRenderer
{
public:
    PostProcessingChainRenderer();

    void applyPostProcessing(const utils::Texture& in_out_hdr_color_buffer);

private:
    std::vector<std::unique_ptr<PostProcessingRenderer>> post_processing_renderers;
};

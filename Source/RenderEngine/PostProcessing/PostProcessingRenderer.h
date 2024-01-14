#pragma once

#include "Models/RawModel.h"
#include "RenderEngine/PostProcessing/Bloom/BloomEffectRenderer.h"

class PostProcessingRenderer
{
public:
    void render(const utils::Texture& hdr_color_buffer, const utils::Texture& output_texture);

private:
    BloomEffectRenderer bloom_effect_renderer;
};

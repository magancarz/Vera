#include "PostProcessingRenderer.h"

void PostProcessingRenderer::render(const utils::Texture& hdr_color_buffer, const utils::Texture& output_texture)
{
    bloom_effect_renderer.apply(hdr_color_buffer, output_texture);
}

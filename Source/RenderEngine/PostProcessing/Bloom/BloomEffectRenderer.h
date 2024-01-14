#pragma once

#include "Models/RawModel.h"
#include "HorizontalBloomShader.h"
#include "VerticalBloomShader.h"
#include "CombineShader.h"
#include "BrightColorsExtractShader.h"

class BloomEffectRenderer
{
public:
    BloomEffectRenderer();

    void apply(const utils::Texture& hdr_color_buffer, const utils::Texture& output_texture);

    utils::Texture color_texture;
    utils::Texture bright_color_texture;

private:
    void prepareShaders();
    void createBrightColorExtractFramebuffer();
    void createBlurFramebuffers();
    void runGaussianBlurIterations();
    void combineHDRColorBufferAndBlurredBrightColorTexture(const utils::Texture& output_texture);

    utils::FBO bloom_fbo;

    utils::FBO ping_pong_fbo1;
    utils::Texture ping_pong_color_buffer1;
    utils::FBO ping_pong_fbo2;
    utils::Texture ping_pong_color_buffer2;

    utils::FBO combine_fbo;

    BrightColorsExtractShader bright_colors_extract_shader;
    HorizontalBloomShader horizontal_bloom_shader;
    VerticalBloomShader vertical_bloom_shader;
    CombineShader combine_shader;
};

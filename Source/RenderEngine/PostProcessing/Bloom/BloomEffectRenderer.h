#pragma once

#include "Models/RawModel.h"
#include "HorizontalBlurShader.h"
#include "VerticalBlurShader.h"
#include "CombineShader.h"
#include "BrightColorsExtractShader.h"
#include "RenderEngine/PostProcessing/PostProcessingRenderer.h"

class BloomEffectRenderer : public PostProcessingRenderer
{
public:
    BloomEffectRenderer();
    ~BloomEffectRenderer() override = default;

    void apply(const utils::Texture& in_out_hdr_color_buffer) override;

private:
    void prepareShaders();
    void createBrightColorExtractFramebuffer();
    void createBlurFramebuffers();
    void extractBrightColors(const utils::Texture& in_out_hdr_color_buffer);
    void runGaussianBlurIterations();
    void combineHDRColorBufferAndBlurredBrightColorTexture(const utils::Texture& in_out_hdr_color_buffer);

    size_t blur_iterations{10};

    utils::FBO bright_colors_extract_fbo;
    utils::Texture color_texture;
    utils::Texture bright_color_texture;

    utils::FBO horizontal_blur_fbo;
    utils::Texture horizontal_blur_color_buffer;
    utils::FBO vertical_blur_fbo;
    utils::Texture vertical_blur_color_buffer;

    utils::FBO combine_fbo;

    BrightColorsExtractShader bright_colors_extract_shader;
    HorizontalBlurShader horizontal_blur_shader;
    VerticalBlurShader vertical_blur_shader;
    CombineShader combine_shader;
};

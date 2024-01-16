#include "BloomEffectRenderer.h"
#include "GUI/Display.h"
#include "RenderEngine/RendererDefines.h"
#include "RenderEngine/RenderingUtils/RenderingUtils.h"

BloomEffectRenderer::BloomEffectRenderer()
{
    prepareShaders();
    createBrightColorExtractFramebuffer();
    createBlurFramebuffers();
}

void BloomEffectRenderer::prepareShaders()
{
    bright_colors_extract_shader.getAllUniformLocations();
    bright_colors_extract_shader.connectTextureUnits();

    horizontal_bloom_shader.getAllUniformLocations();
    horizontal_bloom_shader.connectTextureUnits();

    vertical_bloom_shader.getAllUniformLocations();
    vertical_bloom_shader.connectTextureUnits();

    combine_shader.getAllUniformLocations();
    combine_shader.connectTextureUnits();
}

void BloomEffectRenderer::createBrightColorExtractFramebuffer()
{
    bright_colors_extract_fbo.bindFramebuffer();

    unsigned int color_buffers[2] = {color_texture.texture_id, bright_color_texture.texture_id};
    for (unsigned int i = 0; i < 2; i++)
    {
        glBindTexture(GL_TEXTURE_2D, color_buffers[i]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, Display::WINDOW_WIDTH, Display::WINDOW_HEIGHT, 0, GL_RGBA, GL_FLOAT, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, GL_TEXTURE_2D, color_buffers[i], 0);
    }

    unsigned int attachments[2] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
    glDrawBuffers(2, attachments);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
    {
        std::cout << "Framebuffer not complete!\n";
    }

    bright_colors_extract_fbo.unbind();
}

void BloomEffectRenderer::createBlurFramebuffers()
{
    unsigned int ping_pong_FBO[2] = {horizontal_blur_fbo.FBO_id, vertical_blur_fbo.FBO_id};
    unsigned int ping_pong_color_buffers[2] = {horizontal_blur_color_buffer.texture_id, vertical_blur_color_buffer.texture_id};
    for (size_t i = 0; i < 2; ++i)
    {
        glBindFramebuffer(GL_FRAMEBUFFER, ping_pong_FBO[i]);
        glBindTexture(GL_TEXTURE_2D, ping_pong_color_buffers[i]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, Display::WINDOW_WIDTH, Display::WINDOW_HEIGHT, 0, GL_RGBA, GL_FLOAT, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, ping_pong_color_buffers[i], 0);
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        {
            std::cout << "Framebuffer not complete!\n";
        }
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void BloomEffectRenderer::apply(const utils::Texture& in_out_hdr_color_buffer)
{
    extractBrightColors(in_out_hdr_color_buffer);
    runGaussianBlurIterations();
    combineHDRColorBufferAndBlurredBrightColorTexture(in_out_hdr_color_buffer);
}

void BloomEffectRenderer::extractBrightColors(const utils::Texture& in_out_hdr_color_buffer)
{
    glActiveTexture(GL_TEXTURE0 + RendererDefines::POST_PROCESSING_TEXTURE_INDEX + 0);
    in_out_hdr_color_buffer.bindTexture();
    bright_colors_extract_fbo.bindFramebuffer();
    bright_colors_extract_shader.start();
    RenderingUtils::renderQuad();
    bright_colors_extract_fbo.unbind();
}

void BloomEffectRenderer::runGaussianBlurIterations()
{
    glActiveTexture(GL_TEXTURE0 + RendererDefines::POST_PROCESSING_TEXTURE_INDEX + 0);
    utils::Texture* next_texture_to_bind = &bright_color_texture;
    bool horizontal = true;
    for (size_t i = 0; i < blur_iterations; ++i)
    {
        next_texture_to_bind->bindTexture();

        if (horizontal)
        {
            horizontal_blur_fbo.bindFramebuffer();
            horizontal_bloom_shader.start();
            next_texture_to_bind = &horizontal_blur_color_buffer;
        }
        else
        {
            vertical_blur_fbo.bindFramebuffer();
            vertical_bloom_shader.start();
            next_texture_to_bind = &vertical_blur_color_buffer;
        }

        RenderingUtils::renderQuad();

        horizontal = !horizontal;
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void BloomEffectRenderer::combineHDRColorBufferAndBlurredBrightColorTexture(const utils::Texture& in_out_hdr_color_buffer)
{
    combine_fbo.bindFramebuffer();
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, in_out_hdr_color_buffer.texture_id, 0);

    glActiveTexture(GL_TEXTURE0 + RendererDefines::POST_PROCESSING_TEXTURE_INDEX + 0);
    vertical_blur_color_buffer.bindTexture();

    glActiveTexture(GL_TEXTURE0 + RendererDefines::POST_PROCESSING_TEXTURE_INDEX + 1);
    color_texture.bindTexture();

    combine_shader.start();
    RenderingUtils::renderQuad();

    combine_fbo.unbind();
}
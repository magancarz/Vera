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
    bloom_fbo.bindFramebuffer();

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

    bloom_fbo.unbind();
}

void BloomEffectRenderer::createBlurFramebuffers()
{
    unsigned int ping_pong_FBO[2] = {ping_pong_fbo1.FBO_id, ping_pong_fbo2.FBO_id};
    unsigned int ping_pong_color_buffers[2] = {ping_pong_color_buffer1.texture_id, ping_pong_color_buffer2.texture_id};
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

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }
}

void BloomEffectRenderer::apply(const utils::Texture& hdr_color_buffer, const utils::Texture& output_texture)
{
    glActiveTexture(GL_TEXTURE0 + RendererDefines::POST_PROCESSING_TEXTURE_INDEX + 0);
    hdr_color_buffer.bindTexture();

    bloom_fbo.bindFramebuffer();
    bright_colors_extract_shader.start();
    RenderingUtils::renderQuad();
    bloom_fbo.unbind();

    runGaussianBlurIterations();
    combineHDRColorBufferAndBlurredBrightColorTexture(output_texture);
}

void BloomEffectRenderer::runGaussianBlurIterations()
{
    bool horizontal = true;
    size_t iterations = 10;
    glActiveTexture(GL_TEXTURE0 + RendererDefines::POST_PROCESSING_TEXTURE_INDEX + 0);
    utils::Texture* texture_to_bind = &bright_color_texture;
    for (size_t i = 0; i < iterations; ++i)
    {
        texture_to_bind->bindTexture();

        if (horizontal)
        {
            ping_pong_fbo1.bindFramebuffer();
            horizontal_bloom_shader.start();
            texture_to_bind = &ping_pong_color_buffer1;
        }
        else
        {
            ping_pong_fbo2.bindFramebuffer();
            vertical_bloom_shader.start();
            texture_to_bind = &ping_pong_color_buffer2;
        }

        RenderingUtils::renderQuad();

        horizontal = !horizontal;
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void BloomEffectRenderer::combineHDRColorBufferAndBlurredBrightColorTexture(const utils::Texture& output_texture)
{
    combine_fbo.bindFramebuffer();
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, output_texture.texture_id, 0);

    glActiveTexture(GL_TEXTURE0 + RendererDefines::POST_PROCESSING_TEXTURE_INDEX + 0);
    ping_pong_color_buffer2.bindTexture();

    glActiveTexture(GL_TEXTURE0 + RendererDefines::POST_PROCESSING_TEXTURE_INDEX + 1);
    color_texture.bindTexture();

    combine_shader.start();
    RenderingUtils::renderQuad();

    combine_fbo.unbind();
}
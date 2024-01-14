#include "SSAORenderer.h"

#include "GUI/Display.h"
#include "RenderEngine/RendererDefines.h"
#include "RenderEngine/RenderingUtils/RenderingUtils.h"

SSAORenderer::SSAORenderer()
{
    generateSampleKernel();
    generateNoise();
    createSSAOFramebuffer();
    createSSAOBlurFramebuffer();
    prepareSSAOShader();
}

void SSAORenderer::generateSampleKernel()
{
    ssao_kernel.reserve(64);
    for (size_t i = 0; i < 64; ++i)
    {
        glm::vec3 sample
        {
            random_floats(generator) * 2.0f - 1.0f,
            random_floats(generator) * 2.0f - 1.0f,
            random_floats(generator)
        };
        sample = glm::normalize(sample);
        sample *= random_floats(generator);
        auto scale = static_cast<float>(i) / 60.0f;
        scale = lerp(0.1f, 1.0f, scale * scale);
        sample *= scale;
        ssao_kernel.push_back(sample);
    }
}

float SSAORenderer::lerp(float a, float b, float f)
{
    return a + f * (b - a);
}

void SSAORenderer::generateNoise()
{
    ssao_noise.reserve(16);
    for (size_t i = 0; i < 16; ++i)
    {
        glm::vec3 noise
        {
            random_floats(generator) * 2.0f - 1.0f,
            random_floats(generator) * 2.0f - 1.0f,
            0.0f
        };
        ssao_noise.push_back(noise);
    }

    noise_texture.bindTexture();
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, 4, 4, 0, GL_RGB, GL_FLOAT, &ssao_noise[0]);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
}

void SSAORenderer::createSSAOFramebuffer()
{
    ssao_fbo.bindFramebuffer();
    ssao_color_buffer.bindTexture();
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, Display::WINDOW_WIDTH, Display::WINDOW_HEIGHT, 0, GL_RGB, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, ssao_color_buffer.texture_id, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
    {
        std::cout << "SSAO Framebuffer not complete!\n";
    }

    ssao_fbo.unbind();
}

void SSAORenderer::createSSAOBlurFramebuffer()
{
    ssao_blur_fbo.bindFramebuffer();
    ssao_blur_color_buffer.bindTexture();
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, Display::WINDOW_WIDTH, Display::WINDOW_HEIGHT, 0, GL_RGB, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, ssao_blur_color_buffer.texture_id, 0);

    ssao_blur_fbo.unbind();
}

void SSAORenderer::prepareSSAOShader()
{
    ssao_shader.getAllUniformLocations();
    ssao_shader.connectTextureUnits();
    ssao_shader.loadSamples(ssao_kernel);

    ssao_blur_shader.getAllUniformLocations();
    ssao_blur_shader.connectTextureUnits();
}

void SSAORenderer::render()
{
    glActiveTexture(GL_TEXTURE0 + RendererDefines::G_BUFFER_STARTING_INDEX + RendererDefines::NUMBER_OF_G_BUFFER_TEXTURES + 0);

    ssao_fbo.bindFramebuffer();
    noise_texture.bindTexture();
    ssao_shader.start();
    RenderingUtils::renderQuad();

    ssao_blur_fbo.bindFramebuffer();
    ssao_color_buffer.bindTexture();
    ssao_blur_shader.start();
    RenderingUtils::renderQuad();

    ssao_blur_fbo.unbind();
}
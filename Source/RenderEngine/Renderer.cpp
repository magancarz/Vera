#include "Renderer.h"

#include "GUI/Display.h"
#include "Objects/TriangleMesh.h"
#include "RenderEngine/RenderingUtils/RenderingUtils.h"
#include "RendererDefines.h"

Renderer::Renderer()
{
    createHDRFramebuffer();
    prepareRenderer();
}

void Renderer::createHDRFramebuffer()
{
    hdr_color_buffer.bindTexture();
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F,
                 Display::WINDOW_WIDTH, Display::WINDOW_HEIGHT,
                 0, GL_RGBA, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, hdr_color_buffer.texture_id, 0);

    hdr_rbo_depth.bindRenderbuffer();
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, Display::WINDOW_WIDTH, Display::WINDOW_HEIGHT);

    hdr_fbo.bindFramebuffer();
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, hdr_color_buffer.texture_id, 0);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, hdr_rbo_depth.buffer_id);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
    {
        std::cout << "HDR framebuffer not complete yet!\n";
    }

    hdr_fbo.unbind();
}

void Renderer::prepareRenderer()
{
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_STENCIL_TEST);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glClearColor(0, 0, 0, 1);
    glStencilMask(0x00);
}

void Renderer::renderScene(
        const std::shared_ptr<Camera>& camera,
        const std::vector<std::weak_ptr<Light>>& lights,
        const std::map<std::shared_ptr<RawModel>, std::vector<std::weak_ptr<TriangleMesh>>>& triangle_meshes)
{
    prepareForRendering();
    deferred_shading_renderer.renderSceneObjects(hdr_fbo, triangle_meshes, lights, camera);
    skybox_renderer.renderSkybox(hdr_fbo, camera);
    post_processing_chain_renderer.applyPostProcessing(hdr_color_buffer);
    applyToneMappingAndRenderToDefaultFramebuffer();
}

void Renderer::renderImage(unsigned texture_id)
{
    image_shader.start();
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture_id);
    RenderingUtils::renderYInvertedQuad();
}

void Renderer::prepareForRendering()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
}

void Renderer::applyToneMappingAndRenderToDefaultFramebuffer()
{
    tone_mapping_shader.start();
    glActiveTexture(GL_TEXTURE0 + RendererDefines::HDR_BUFFER_INDEX);
    hdr_color_buffer.bindTexture();
    RenderingUtils::renderQuad();
}

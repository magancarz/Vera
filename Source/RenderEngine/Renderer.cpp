#include "Renderer.h"

#include "GUI/Display.h"
#include "Objects/TriangleMesh.h"
#include "RenderEngine/RenderingUtils/RenderingUtils.h"
#include "RendererDefines.h"

Renderer::Renderer()
{
    ray_traced_image_shader.getAllUniformLocations();
    ray_traced_image_shader.connectTextureUnits();
    createHDRFramebuffer();
}

void Renderer::createHDRFramebuffer()
{
    hdr_color_texture.bindTexture();
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F,
                 Display::WINDOW_WIDTH, Display::WINDOW_HEIGHT,
                 0, GL_RGBA, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, hdr_color_texture.texture_id, 0);

    hdr_rbo_depth.bindRenderbuffer();
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, Display::WINDOW_WIDTH, Display::WINDOW_HEIGHT);

    hdr_fbo.bindFramebuffer();
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, hdr_color_texture.texture_id, 0);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, hdr_rbo_depth.buffer_id);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
    {
        std::cout << "Framebuffer not complete!\n";
    }

    hdr_fbo.unbind();

    hdr_shader.getAllUniformLocations();
    hdr_shader.connectTextureUnits();
}

void Renderer::renderScene(const std::shared_ptr<Camera>& camera, const std::vector<std::weak_ptr<Light>>& lights, const std::vector<std::weak_ptr<TriangleMesh>>& entities)
{
    prepare();
    processEntities(entities);
    deferred_shading_renderer.renderSceneObjects(hdr_fbo, objects_map, lights, camera);
    skybox_renderer.renderSkybox(hdr_fbo, camera);
    bloom_effect_renderer.apply(hdr_color_texture, hdr_color_texture);
    applyToneMappingAndRenderToDefaultFramebuffer();
    cleanUpObjectsMaps();
}

void Renderer::renderRayTracedImage(unsigned texture_id) const
{
    ray_traced_image_shader.start();
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture_id);
    RenderingUtils::renderYInvertedQuad();
}

void Renderer::prepare()
{
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_STENCIL_TEST);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glClearColor(0, 0, 0, 1);
    glStencilMask(0xFF);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
}

void Renderer::processEntities(const std::vector<std::weak_ptr<TriangleMesh>>& entities)
{
    for (const auto& entity : entities)
    {
        processEntity(objects_map, entity);
    }
}

void Renderer::processEntity(std::map<std::shared_ptr<RawModel>, std::vector<std::weak_ptr<TriangleMesh>>>& map, const std::weak_ptr<TriangleMesh>& entity)
{
    auto entity_model = entity.lock()->getModelData();

    const auto it = map.find(entity_model);
    if (it != map.end())
    {
        auto& batch = it->second;
        batch.push_back(entity);
    }
    else
    {
        std::vector<std::weak_ptr<TriangleMesh>> new_batch;
        new_batch.push_back(entity);
        map.insert(std::make_pair(entity_model, new_batch));
    }
}

void Renderer::applyToneMappingAndRenderToDefaultFramebuffer()
{
    hdr_shader.start();
    glActiveTexture(GL_TEXTURE0 + RendererDefines::HDR_BUFFER_INDEX);
    hdr_color_texture.bindTexture();
    RenderingUtils::renderQuad();
}

void Renderer::cleanUpObjectsMaps()
{
    objects_map.clear();
}

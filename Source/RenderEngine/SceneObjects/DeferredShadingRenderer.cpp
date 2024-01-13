#include "DeferredShadingRenderer.h"

#include "RenderEngine/Camera.h"
#include "RenderEngine/RendererDefines.h"
#include "GUI/Display.h"
#include "Materials/Material.h"

DeferredShadingRenderer::DeferredShadingRenderer()
{
    prepareSceneObjectsRenderers();
    createGBuffer();
}

void DeferredShadingRenderer::prepareSceneObjectsRenderers()
{
    for (const auto& scene_object_renderer : scene_objects_renderers)
    {
        scene_object_renderer->prepare();
        scene_object_renderer->bindUniformBuffer(transformation_matrices_uniform_buffer);
    }

    outline_mark_shader.bindUniformBuffer(transformation_matrices_uniform_buffer);

    outline_shader.getAllUniformLocations();
    outline_shader.loadOutlineColor(glm::vec3{0, 1, 1});

    lighting_pass_renderer.bindUniformBuffer(light_info_uniform_buffer);

    light_objects_shader.getAllUniformLocations();
    light_objects_shader.bindUniformBuffer(transformation_matrices_uniform_buffer);
}

void DeferredShadingRenderer::createGBuffer()
{
    g_buffer.bindFramebuffer();

    g_position.bindTexture();
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F,
         Display::WINDOW_WIDTH, Display::WINDOW_HEIGHT,
         0, GL_RGBA, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, g_position.texture_id, 0);

    g_normal.bindTexture();
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F,
         Display::WINDOW_WIDTH, Display::WINDOW_HEIGHT,
         0, GL_RGBA, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, g_normal.texture_id, 0);

    g_color_spec.bindTexture();
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, Display::WINDOW_WIDTH, Display::WINDOW_HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, g_color_spec.texture_id, 0);

    unsigned int attachments[3] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2};
    glDrawBuffers(3, attachments);

    rbo_depth.bindRenderbuffer();
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, Display::WINDOW_WIDTH, Display::WINDOW_HEIGHT);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rbo_depth.buffer_id);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
    {
        std::cout << "Framebuffer not complete!" << std::endl;
    }

    g_buffer.unbind();
}

void DeferredShadingRenderer::renderSceneObjects(
        const std::map<std::shared_ptr<RawModel>, std::vector<std::weak_ptr<TriangleMesh>>>& entity_map,
        const std::vector<std::weak_ptr<Light>>& lights, const std::shared_ptr<Camera>& camera)
{
    prepareLights(lights);
    prepareTransformationMatrices(camera);

    renderShadowMap(entity_map, lights);

    std::map<std::shared_ptr<RawModel>, std::vector<std::shared_ptr<TriangleMesh>>> light_objects;

    g_buffer.bindFramebuffer();
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

    glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);
    glStencilFunc(GL_ALWAYS, 0, 0x00);
    glStencilMask(0xFF);
    for (const auto& [raw_model, entities] : entity_map)
    {
        raw_model->prepareModel();
        for (const auto& entity : entities)
        {
            auto entity_shared_ptr = entity.lock();
            if (entity_shared_ptr->getMaterial()->isEmittingLight()) {
                light_objects[raw_model].push_back(entity_shared_ptr);
                continue;
            }
            std::shared_ptr<SceneObjectsRenderer> applied_renderer;
            for (const auto& scene_object_renderer : scene_objects_renderers)
            {
                if (scene_object_renderer->apply(entity_shared_ptr))
                {
                    applied_renderer = scene_object_renderer;
                    break;
                }
            }
            applied_renderer->prepareInstance(entity_shared_ptr);

            loadTransformationMatrix(entity_shared_ptr->getTransform());
            if (entity_shared_ptr->shouldBeOutlined())
            {
                glStencilFunc(GL_ALWAYS, 1, 0xFF);
                glStencilMask(0xFF);
                glDisable(GL_DEPTH_TEST);
                outline_mark_shader.start();
                glDrawElements(GL_TRIANGLES, static_cast<int>(raw_model->vertex_count), GL_UNSIGNED_INT, nullptr);
                glEnable(GL_DEPTH_TEST);
                ShaderProgram::stop();
            }
            glStencilFunc(GL_ALWAYS, 0, 0x00);

            applied_renderer->prepareShader();
            glDrawElements(GL_TRIANGLES, static_cast<int>(raw_model->vertex_count), GL_UNSIGNED_INT, nullptr);
        }
        raw_model->unbindModel();
    }

    g_buffer.unbind();

    lighting_pass_renderer.render(camera, g_position, g_normal, g_color_spec);

    glBindFramebuffer(GL_READ_FRAMEBUFFER, g_buffer.FBO_id);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
    glBlitFramebuffer(0, 0, Display::WINDOW_WIDTH, Display::WINDOW_HEIGHT,
                      0, 0, Display::WINDOW_WIDTH, Display::WINDOW_HEIGHT, GL_DEPTH_BUFFER_BIT, GL_NEAREST);
    glBlitFramebuffer(0, 0, Display::WINDOW_WIDTH, Display::WINDOW_HEIGHT,
                      0, 0, Display::WINDOW_WIDTH, Display::WINDOW_HEIGHT, GL_STENCIL_BUFFER_BIT, GL_NEAREST);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    renderLightObjects(light_objects);
    renderOutlines();
}

void DeferredShadingRenderer::prepareLights(const std::vector<std::weak_ptr<Light>>& lights)
{
    bindShadowMaps(lights);

    for (const auto& light : lights)
    {
        auto light_as_shared_ptr = light.lock();
        LightInfo light_info
        {
            light_as_shared_ptr->getPosition(),
            light_as_shared_ptr->getLightColor(),
            light_as_shared_ptr->getAttenuation(),
        };
        light_info_uniform_buffer.setValue(light_info);
    }
}

void DeferredShadingRenderer::bindShadowMaps(const std::vector<std::weak_ptr<Light>>& lights)
{
    GLenum texture_binding = GL_TEXTURE0 + RendererDefines::SHADOW_MAPS_TEXTURES_STARTING_INDEX;
    for (const auto& light : lights)
    {
        glActiveTexture(texture_binding);
        light.lock()->getShadowMap()->bindTexture();
        ++texture_binding;
    }
}

void DeferredShadingRenderer::prepareTransformationMatrices(const std::shared_ptr<Camera>& camera) const
{
    auto view = camera->getCameraViewMatrix();
    auto proj = camera->getPerspectiveProjectionMatrix();
    transformation_matrices_uniform_buffer.setValue({glm::mat4{}, view, proj});
}

void DeferredShadingRenderer::loadTransformationMatrix(const glm::mat4& model) const
{
    transformation_matrices_uniform_buffer.setValue(model, 0);
}

void DeferredShadingRenderer::renderShadowMap(
        const std::map<std::shared_ptr<RawModel>, std::vector<std::weak_ptr<TriangleMesh>>>& entity_map,
        const std::vector<std::weak_ptr<Light>>& lights)
{
    shadow_map_renderer.renderSceneToDepthBuffers(entity_map, lights);
}

void DeferredShadingRenderer::renderLightObjects(
        const std::map<std::shared_ptr<RawModel>, std::vector<std::shared_ptr<TriangleMesh>>>& light_objects)
{
    light_objects_shader.start();
    for (const auto& [raw_model, entities] : light_objects)
    {
        raw_model->prepareModel();
        for (auto& entity : entities)
        {
            light_objects_shader.loadLightColor(entity->getMaterial()->getColor());
            transformation_matrices_uniform_buffer.setValue(entity->getTransform(), 0);
            glDrawElements(GL_TRIANGLES, static_cast<int>(raw_model->vertex_count), GL_UNSIGNED_INT, nullptr);
        }
        raw_model->unbindModel();
    }
}

void DeferredShadingRenderer::renderOutlines()
{
    glStencilFunc(GL_EQUAL, 1, 0xFF);

    glBindVertexArray(AssetManager::texture_quad.vao->vao_id);
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);

    glDisable(GL_DEPTH_TEST);
    outline_shader.start();
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glEnable(GL_DEPTH_TEST);

    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
    glBindVertexArray(0);
}
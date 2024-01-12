#include "DeferredShadingRenderer.h"

#include "RenderEngine/Camera.h"
#include "RenderEngine/RendererDefines.h"
#include "GUI/Display.h"

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

    outline_shader.getAllUniformLocations();
    outline_shader.loadOutlineColor(glm::vec3{0, 1, 1});
    outline_shader.bindUniformBuffer(transformation_matrices_uniform_buffer);

    lighting_pass_renderer.bindUniformBuffer(light_info_uniform_buffer);
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
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, Display::WINDOW_WIDTH, Display::WINDOW_HEIGHT);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rbo_depth.buffer_id);
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

    g_buffer.bindFramebuffer();
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    for (const auto& [raw_model, entities] : entity_map)
    {
        prepareTexturedModel(raw_model);
        for (const auto& entity : entities)
        {
            auto entity_shared_ptr = entity.lock();
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
            applied_renderer->prepareShader();
            glDrawElements(GL_TRIANGLES, static_cast<int>(raw_model->vertex_count), GL_UNSIGNED_INT, nullptr);
        }
        unbindTexturedModel();
    }

    g_buffer.unbind();

    lighting_pass_renderer.render(g_position, g_normal, g_color_spec);
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

void DeferredShadingRenderer::prepareTexturedModel(const std::shared_ptr<RawModel>& raw_model) const
{
    glBindVertexArray(raw_model->vao->vao_id);
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);
    glEnableVertexAttribArray(3);
    glEnableVertexAttribArray(4);
}

void DeferredShadingRenderer::unbindTexturedModel()
{
    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(2);
    glDisableVertexAttribArray(3);
    glDisableVertexAttribArray(4);
    glBindVertexArray(0);
}
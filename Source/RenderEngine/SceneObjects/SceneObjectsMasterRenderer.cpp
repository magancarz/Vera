#include "SceneObjectsMasterRenderer.h"

#include "RenderEngine/Camera.h"
#include "RenderEngine/RendererDefines.h"

SceneObjectsMasterRenderer::SceneObjectsMasterRenderer()
{
    prepareSceneObjectsRenderers();
}

void SceneObjectsMasterRenderer::prepareSceneObjectsRenderers()
{
    for (const auto& scene_object_renderer : scene_objects_renderers)
    {
        scene_object_renderer->prepare();
        scene_object_renderer->bindUniformBuffer(light_info_uniform_buffer);
        scene_object_renderer->bindUniformBuffer(transformation_matrices_uniform_buffer);
    }

    outline_shader.getAllUniformLocations();
    outline_shader.loadOutlineColor(glm::vec3{0, 1, 1});
    outline_shader.bindUniformBuffer(transformation_matrices_uniform_buffer);
}

void SceneObjectsMasterRenderer::renderSceneObjects(
        const std::map<std::shared_ptr<RawModel>, std::vector<std::weak_ptr<TriangleMesh>>>& entity_map,
        const std::vector<std::weak_ptr<Light>>& lights, const std::shared_ptr<Camera>& camera)
{
    prepareLights(lights);
    prepareTransformationMatrices(camera);

    renderShadowMap(entity_map, lights);

    glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);
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

            if (entity_shared_ptr->shouldBeOutlined())
            {
                glStencilFunc(GL_ALWAYS, 1, 0xFF);
                glStencilMask(0xFF);
            }
            else
            {
                glStencilMask(0x00);
            }

            applied_renderer->prepareShader();
            glDrawElements(GL_TRIANGLES, static_cast<int>(raw_model->vertex_count), GL_UNSIGNED_INT, nullptr);

            if (entity_shared_ptr->shouldBeOutlined())
            {
                glStencilFunc(GL_NOTEQUAL, 1, 0xFF);
                glStencilMask(0x00);
                outline_shader.start();
                glDrawElements(GL_TRIANGLES, static_cast<int>(raw_model->vertex_count), GL_UNSIGNED_INT, nullptr);
                glStencilFunc(GL_ALWAYS, 1, 0x00);
                glStencilMask(0xFF);
                ShaderProgram::stop();
            }
        }
        unbindTexturedModel();
    }
}

void SceneObjectsMasterRenderer::prepareLights(const std::vector<std::weak_ptr<Light>>& lights)
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

void SceneObjectsMasterRenderer::bindShadowMaps(const std::vector<std::weak_ptr<Light>>& lights)
{
    GLenum texture_binding = GL_TEXTURE0 + RendererDefines::SHADOW_MAPS_TEXTURES_STARTING_INDEX;
    for (const auto& light : lights)
    {
        glActiveTexture(texture_binding);
        light.lock()->getShadowMap()->bindTexture();
        ++texture_binding;
    }
}

void SceneObjectsMasterRenderer::prepareTransformationMatrices(const std::shared_ptr<Camera>& camera) const
{
    auto view = camera->getCameraViewMatrix();
    auto proj = camera->getPerspectiveProjectionMatrix();
    transformation_matrices_uniform_buffer.setValue({glm::mat4{}, view, proj});
}

void SceneObjectsMasterRenderer::loadTransformationMatrix(const glm::mat4& model) const
{
    transformation_matrices_uniform_buffer.setValue(model, 0);
}

void SceneObjectsMasterRenderer::renderShadowMap(
        const std::map<std::shared_ptr<RawModel>, std::vector<std::weak_ptr<TriangleMesh>>>& entity_map,
        const std::vector<std::weak_ptr<Light>>& lights)
{
    shadow_map_renderer.renderSceneToDepthBuffers(entity_map, lights);
}

void SceneObjectsMasterRenderer::prepareTexturedModel(const std::shared_ptr<RawModel>& raw_model) const
{
    glBindVertexArray(raw_model->vao->vao_id);
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);
    glEnableVertexAttribArray(3);
    glEnableVertexAttribArray(4);
}

void SceneObjectsMasterRenderer::unbindTexturedModel()
{
    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(2);
    glDisableVertexAttribArray(3);
    glDisableVertexAttribArray(4);
    glBindVertexArray(0);
}
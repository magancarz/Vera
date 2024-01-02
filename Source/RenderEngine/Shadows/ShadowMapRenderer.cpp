#include "ShadowMapRenderer.h"

#include <GL/glew.h>

#include "GUI/Display.h"
#include "Objects/TriangleMesh.h"
#include "Objects/Lights/Light.h"
#include "Objects/Lights/DirectionalLight.h"

ShadowMapRenderer::ShadowMapRenderer()
{
    createShadowMapFrameBuffer();
}

void ShadowMapRenderer::createShadowMapFrameBuffer()
{
    glBindFramebuffer(GL_FRAMEBUFFER, depth_map_FBO.FBO_id);
    glDrawBuffer(GL_NONE);
    glReadBuffer(GL_NONE);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void ShadowMapRenderer::renderSceneToDepthBuffers(
        const std::map<std::shared_ptr<RawModel>, std::vector<std::weak_ptr<TriangleMesh>>>& entity_map,
        const std::vector<std::weak_ptr<Light>>& lights)
{
    glBindFramebuffer(GL_FRAMEBUFFER, depth_map_FBO.FBO_id);

    for (const auto& light : lights)
    {
        if (auto temp = dynamic_cast<DirectionalLight*>(light.lock().get()))
        {
            light.lock()->prepareForShadowMapRendering();
            draw(entity_map, light);
        }
        ShaderProgram::stop();
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, Display::WINDOW_WIDTH, Display::WINDOW_HEIGHT);
}

void ShadowMapRenderer::draw(
        const std::map<std::shared_ptr<RawModel>, std::vector<std::weak_ptr<TriangleMesh>>>& entity_map,
        const std::weak_ptr<Light>& light)
{
    glClear(GL_DEPTH_BUFFER_BIT);

    for (const auto& [raw_model, entities] : entity_map)
    {
        prepareTexturedModel(raw_model);

        for (const auto& entity : entities)
        {
            prepareInstance(entity, light);
            glDrawElements(GL_TRIANGLES, static_cast<int>(raw_model->vertex_count), GL_UNSIGNED_INT, nullptr);
        }

        unbindTexturedModel();
    }
}

void ShadowMapRenderer::prepareTexturedModel(const std::shared_ptr<RawModel>& raw_model) const
{
    glBindVertexArray(raw_model->vao->vao_id);
    glEnableVertexAttribArray(0);
}

void ShadowMapRenderer::unbindTexturedModel()
{
    glDisableVertexAttribArray(0);
    glBindVertexArray(0);
}

void ShadowMapRenderer::prepareInstance(const std::weak_ptr<TriangleMesh>& entity, const std::weak_ptr<Light>& light) const
{
    const auto entity_rotation = entity.lock()->getRotation();
    const auto transformation_matrix = Algorithms::createTransformationMatrix
    (
            entity.lock()->getPosition(),
            entity_rotation.x,
            entity_rotation.y,
            entity_rotation.z,
            entity.lock()->getScale()
    );

    light.lock()->loadTransformationMatrixToShadowMapShader(transformation_matrix);
}

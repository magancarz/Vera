#include "EntityRenderer.h"

#include <GL/glew.h>

#include "RenderEngine/Renderer.h"
#include "Utils/Algorithms.h"
#include "Objects/Object.h"
#include "Objects/TriangleMesh.h"
#include "Materials/Material.h"
#include "Objects/Lights/Light.h"

EntityRenderer::EntityRenderer()
{
    static_shader.start();
    static_shader.bindAttributes();
    static_shader.getAllUniformLocations();
    static_shader.connectTextureUnits();
    ShaderProgram::stop();

    outline_shader.start();
    outline_shader.bindAttributes();
    outline_shader.getAllUniformLocations();
    outline_shader.connectTextureUnits();
    ShaderProgram::stop();
}

void EntityRenderer::render(
    const std::map<std::shared_ptr<RawModel>, std::vector<std::weak_ptr<TriangleMesh>>>& entity_map,
    const std::vector<std::weak_ptr<Light>>& lights,
    const std::shared_ptr<Camera>& camera) const
{
    static_shader.start();
    static_shader.loadLights(lights);
    static_shader.loadViewMatrix(camera);
    static_shader.loadProjectionMatrix(camera);
    ShaderProgram::stop();

    outline_shader.start();
    outline_shader.loadViewMatrix(camera);
    outline_shader.loadProjectionMatrix(camera);
    ShaderProgram::stop();

    glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);

    for (const auto& [raw_model, entities] : entity_map)
    {
        prepareTexturedModel(raw_model);

        for (const auto& entity : entities)
        {
            prepareInstance(entity);

            if (entity.lock()->shouldBeOutlined())
            {
                glStencilFunc(GL_ALWAYS, 1, 0xFF);
                glStencilMask(0xFF);
            }
            else
            {
                glStencilMask(0x00);
            }
            static_shader.start();
            glDrawElements(GL_TRIANGLES, static_cast<int>(raw_model->vertex_count), GL_UNSIGNED_INT, nullptr);
            ShaderProgram::stop();

            if (entity.lock()->shouldBeOutlined())
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

void EntityRenderer::prepareTexturedModel(const std::shared_ptr<RawModel>& raw_model) const
{
    glBindVertexArray(raw_model->vao->vao_id);
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);
    glEnableVertexAttribArray(3);
    glEnableVertexAttribArray(4);
}

void EntityRenderer::unbindTexturedModel()
{
    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(2);
    glDisableVertexAttribArray(3);
    glDisableVertexAttribArray(4);
    glBindVertexArray(0);
}

void EntityRenderer::prepareInstance(const std::weak_ptr<TriangleMesh>& entity) const
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

    static_shader.start();
    static_shader.loadTransformationMatrix(transformation_matrix);
    static_shader.loadReflectivity(1.f - entity.lock()->getMaterial()->getFuzziness());
    glActiveTexture(GL_TEXTURE0);
    entity.lock()->getMaterial()->bindColorTexture();
    if (entity.lock()->getMaterial()->hasNormalMap())
    {
        glActiveTexture(GL_TEXTURE1);
        entity.lock()->getMaterial()->bindNormalMap();
        static_shader.loadNormalMapLoadedBool(true);
    }
    else
    {
        static_shader.loadNormalMapLoadedBool(false);
    }

    ShaderProgram::stop();

    if (entity.lock()->shouldBeOutlined())
    {
        const auto scaled_transformation_matrix = Algorithms::createTransformationMatrix
        (
            entity.lock()->getPosition(),
            entity_rotation.x,
            entity_rotation.y,
            entity_rotation.z,
            entity.lock()->getScale() * 1.02f
        );

        outline_shader.start();
        outline_shader.loadTransformationMatrix(scaled_transformation_matrix);
        ShaderProgram::stop();
    }
}
#include "EntityRenderer.h"

#include <GL/glew.h>

#include "RenderEngine/Renderer.h"
#include "Utils/AdditionalAlgorithms.h"
#include "Objects/Object.h"
#include "Materials/Material.h"

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
    const std::map<std::shared_ptr<RawModel>, std::vector<std::shared_ptr<Object>>>& entity_map,
    const std::shared_ptr<Camera>& camera) const
{
    static_shader.start();
    static_shader.loadLights(entity_map);
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

            if (entity->shouldBeOutlined())
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

            if (entity->shouldBeOutlined())
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
}

void EntityRenderer::unbindTexturedModel()
{
    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(2);
    glBindVertexArray(0);
}

void EntityRenderer::prepareInstance(const std::shared_ptr<Object>& entity) const
{
    const auto entity_rotation = entity->getRotation();
    const auto transformation_matrix = AdditionalAlgorithms::createTransformationMatrix
    (
        entity->getPosition(),
        entity_rotation.x,
        entity_rotation.y,
        entity_rotation.z,
        entity->getScale()
    );

    static_shader.start();
    static_shader.loadTransformationMatrix(transformation_matrix);
    glActiveTexture(GL_TEXTURE0);
    entity->getMaterial()->bindColorTexture();
    if (entity->getMaterial()->hasNormalMap())
    {
        glActiveTexture(GL_TEXTURE1);
        entity->getMaterial()->bindNormalMap();
    }

    ShaderProgram::stop();

    if (entity->shouldBeOutlined())
    {
        const auto scaled_transformation_matrix = AdditionalAlgorithms::createTransformationMatrix
        (
            entity->getPosition(),
            entity_rotation.x,
            entity_rotation.y,
            entity_rotation.z,
            entity->getScale() * 1.02f
        );

        outline_shader.start();
        outline_shader.loadTransformationMatrix(scaled_transformation_matrix);
        ShaderProgram::stop();
    }
}
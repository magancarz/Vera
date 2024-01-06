#include "SceneObjectsRenderer.h"

#include "GL/glew.h"

#include "RenderEngine/Renderer.h"
#include "Utils/Algorithms.h"
#include "Objects/TriangleMesh.h"
#include "Materials/Material.h"
#include "Objects/Lights/Light.h"

void SceneObjectsRenderer::prepare()
{
    createSceneObjectShader();
}

void SceneObjectsRenderer::createSceneObjectShader()
{
    static_shader = std::make_unique<SceneObjectsShader>();
    static_shader->getAllUniformLocations();
}

bool SceneObjectsRenderer::apply()
{
    return true;
}

void SceneObjectsRenderer::render(
    const std::map<std::shared_ptr<RawModel>, std::vector<std::weak_ptr<TriangleMesh>>>& entity_map,
    const std::vector<std::weak_ptr<Light>>& lights,
    const std::shared_ptr<Camera>& camera)
{
    static_shader->start();
    static_shader->loadLights(lights);
    static_shader->loadViewAndProjectionMatrices(camera);
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
            static_shader->start();
            glDrawElements(GL_TRIANGLES, static_cast<int>(raw_model->vertex_count), GL_UNSIGNED_INT, nullptr);
            ShaderProgram::stop();

            if (entity.lock()->shouldBeOutlined())
            {
                glStencilFunc(GL_NOTEQUAL, 1, 0xFF);
                glStencilMask(0x00);
//                outline_shader.start();
                glDrawElements(GL_TRIANGLES, static_cast<int>(raw_model->vertex_count), GL_UNSIGNED_INT, nullptr);
                glStencilFunc(GL_ALWAYS, 1, 0x00);
                glStencilMask(0xFF);
                ShaderProgram::stop();
            }
        }

        unbindTexturedModel();
    }
}

void SceneObjectsRenderer::prepareTexturedModel(const std::shared_ptr<RawModel>& raw_model) const
{
    glBindVertexArray(raw_model->vao->vao_id);
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);
}

void SceneObjectsRenderer::unbindTexturedModel()
{
    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(2);
    glBindVertexArray(0);
}

void SceneObjectsRenderer::prepareInstance(const std::weak_ptr<TriangleMesh>& entity)
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

    static_shader->start();
    static_shader->loadTransformationMatrix(transformation_matrix);
    static_shader->loadReflectivity(1.f - entity.lock()->getMaterial()->getFuzziness());
    glActiveTexture(GL_TEXTURE0);
    entity.lock()->getMaterial()->bindColorTexture();

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

//        outline_shader.start();
//        outline_shader.loadTransformationMatrix(scaled_transformation_matrix);
        ShaderProgram::stop();
    }
}
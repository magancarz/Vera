#include "ParallaxMappedSceneObjectsRenderer.h"

#include "GL/glew.h"

#include "RenderEngine/Renderer.h"
#include "Utils/Algorithms.h"
#include "Objects/TriangleMesh.h"
#include "Materials/Material.h"
#include "Objects/Lights/Light.h"

ParallaxMappedSceneObjectsRenderer::ParallaxMappedSceneObjectsRenderer()
{
    parallax_mapped_scene_objects_shader.start();
    parallax_mapped_scene_objects_shader.getAllUniformLocations();
    ShaderProgram::stop();

    outline_shader.start();
    outline_shader.getAllUniformLocations();
    ShaderProgram::stop();
}

void ParallaxMappedSceneObjectsRenderer::render(
    const std::map<std::shared_ptr<RawModel>, std::vector<std::weak_ptr<TriangleMesh>>>& entity_map,
    const std::vector<std::weak_ptr<Light>>& lights,
    const std::shared_ptr<Camera>& camera)
{
    parallax_mapped_scene_objects_shader.start();
    parallax_mapped_scene_objects_shader.loadLights(lights);
    parallax_mapped_scene_objects_shader.loadViewMatrix(camera);
    parallax_mapped_scene_objects_shader.loadProjectionMatrix(camera);

    current_texture = GL_TEXTURE0;
    size_t shadow_map_index = 0;
    size_t cube_shadow_map_index = 0;
    for (const auto& light : lights)
    {
        glActiveTexture(current_texture);
        light.lock()->bindShadowMapTexture();
        if (auto is_point_light = dynamic_cast<PointLight*>(light.lock().get()))
        {
            parallax_mapped_scene_objects_shader.loadTextureIndexToCubeShadowMap(cube_shadow_map_index, shadow_map_index + cube_shadow_map_index);
            ++cube_shadow_map_index;
        }
        ++current_texture;
    }
    parallax_mapped_scene_objects_shader.connectTextureUnits(current_texture);

    parallax_mapped_scene_objects_shader.loadLightsCount(lights.size());
    ShaderProgram::stop();

    outline_shader.start();
    outline_shader.loadViewAndProjectionMatrices(camera);
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
            parallax_mapped_scene_objects_shader.start();
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

void ParallaxMappedSceneObjectsRenderer::prepareTexturedModel(const std::shared_ptr<RawModel>& raw_model) const
{
    glBindVertexArray(raw_model->vao->vao_id);
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);
    glEnableVertexAttribArray(3);
    glEnableVertexAttribArray(4);
}

void ParallaxMappedSceneObjectsRenderer::unbindTexturedModel()
{
    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(2);
    glDisableVertexAttribArray(3);
    glDisableVertexAttribArray(4);
    glBindVertexArray(0);
}

void ParallaxMappedSceneObjectsRenderer::prepareInstance(const std::weak_ptr<TriangleMesh>& entity)
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

    parallax_mapped_scene_objects_shader.start();
    parallax_mapped_scene_objects_shader.loadTransformationMatrix(transformation_matrix);
    parallax_mapped_scene_objects_shader.loadReflectivity(1.f - entity.lock()->getMaterial()->getFuzziness());
    parallax_mapped_scene_objects_shader.loadHeightScale(0.01f);
    glActiveTexture(current_texture + 0);
    entity.lock()->getMaterial()->bindColorTexture();

    glActiveTexture(current_texture + 1);
    entity.lock()->getMaterial()->bindNormalMap();

    glActiveTexture(current_texture + 2);
    entity.lock()->getMaterial()->bindDepthMap();

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
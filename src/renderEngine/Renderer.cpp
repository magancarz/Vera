#include "Renderer.h"

#include "GUI/Display.h"
#include "Objects/TriangleMesh.h"

Renderer::Renderer()
{
    quad = AssetManager::loadSimpleModel(quad_positions, quad_textures);
    ray_traced_image_shader.start();
    ray_traced_image_shader.bindAttributes();
    ray_traced_image_shader.getAllUniformLocations();
    ray_traced_image_shader.connectTextureUnits();
    ShaderProgram::stop();

    entity_renderer = std::make_unique<EntityRenderer>();
}

void Renderer::renderScene(const std::shared_ptr<Camera>& camera, const std::vector<std::weak_ptr<Light>>& lights, const std::vector<std::weak_ptr<TriangleMesh>>& entities)
{
    prepare();
    processEntities(entities);
    entity_renderer->render(entities_map, lights, camera);
    cleanUpObjectsMaps();
}

void Renderer::renderRayTracedImage(unsigned texture_id) const
{
    glBindVertexArray(quad.vao->vao_id);
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glDisable(GL_DEPTH_TEST);

    ray_traced_image_shader.start();
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture_id);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    glEnable(GL_DEPTH_TEST);
    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
    glBindVertexArray(0);
}

void Renderer::prepare()
{
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_STENCIL_TEST);
    glClearColor(0, 0, 0, 1);
    glStencilMask(0xFF);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
}

void Renderer::processEntities(const std::vector<std::weak_ptr<TriangleMesh>>& entities)
{
    for (const auto& entity : entities)
    {
        processEntity(entity);
    }
}

void Renderer::processEntity(const std::weak_ptr<TriangleMesh>& entity)
{
    auto entity_model = entity.lock()->getModelData();

    const auto it = entities_map.find(entity_model);
    if (it != entities_map.end())
    {
        auto& batch = it->second;
        batch.push_back(entity);
    }
    else
    {
        std::vector<std::weak_ptr<TriangleMesh>> new_batch;
        new_batch.push_back(entity);
        entities_map.insert(std::make_pair(entity_model, new_batch));
    }
}

void Renderer::cleanUpObjectsMaps()
{
    entities_map.clear();
}

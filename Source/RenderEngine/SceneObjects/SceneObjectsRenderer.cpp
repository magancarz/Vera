#include "SceneObjectsRenderer.h"

#include "GL/glew.h"

#include "RenderEngine/RendererDefines.h"
#include "Utils/Algorithms.h"
#include "Objects/TriangleMesh.h"
#include "Materials/Material.h"

void SceneObjectsRenderer::prepare()
{
    createSceneObjectShader();
}

void SceneObjectsRenderer::createSceneObjectShader()
{
    scene_object_shader = std::make_unique<SceneObjectsShader>();
    scene_object_shader->getAllUniformLocations();
    scene_object_shader->connectTextureUnits();
}

bool SceneObjectsRenderer::apply()
{
    return true;
}

void SceneObjectsRenderer::prepareTexturedModel(const std::shared_ptr<RawModel>& raw_model) const
{
    glBindVertexArray(raw_model->vao->vao_id);
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);
}

void SceneObjectsRenderer::prepareInstance(const std::weak_ptr<TriangleMesh>& entity)
{
    auto material = entity.lock()->getMaterial();

    glActiveTexture(GL_TEXTURE0 + RendererDefines::MODEL_TEXTURES_STARTING_INDEX);
    material->bindColorTexture();

    scene_object_shader->loadReflectivity(1.f - entity.lock()->getMaterial()->getFuzziness());
}

void SceneObjectsRenderer::prepareShader()
{
    scene_object_shader->start();
}

void SceneObjectsRenderer::unbindTexturedModel()
{
    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(2);
    glBindVertexArray(0);
}
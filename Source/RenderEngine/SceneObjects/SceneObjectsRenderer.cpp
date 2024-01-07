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

bool SceneObjectsRenderer::apply(const std::shared_ptr<TriangleMesh>& entity)
{
    return true;
}

GLenum SceneObjectsRenderer::prepareInstance(const std::shared_ptr<TriangleMesh>& entity)
{
    auto material = entity->getMaterial();

    glActiveTexture(GL_TEXTURE0 + RendererDefines::MODEL_TEXTURES_STARTING_INDEX + 0);
    material->bindColorTexture();

    scene_object_shader->loadReflectivity(1.f - entity->getMaterial()->getFuzziness());

    return GL_TEXTURE0 + RendererDefines::MODEL_TEXTURES_STARTING_INDEX + 1;
}

void SceneObjectsRenderer::prepareShader()
{
    scene_object_shader->start();
}
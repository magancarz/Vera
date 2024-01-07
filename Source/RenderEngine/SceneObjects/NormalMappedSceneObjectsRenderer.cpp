#include "NormalMappedSceneObjectsRenderer.h"

#include "GL/glew.h"

#include "Objects/TriangleMesh.h"
#include "Materials/Material.h"

void NormalMappedSceneObjectsRenderer::createSceneObjectShader()
{
    scene_object_shader = std::make_unique<NormalMappedSceneObjectsShader>();
    scene_object_shader->getAllUniformLocations();
    scene_object_shader->connectTextureUnits();
}

bool NormalMappedSceneObjectsRenderer::apply(const std::shared_ptr<TriangleMesh>& entity)
{
    return entity->isNormalMapped();
}

GLenum NormalMappedSceneObjectsRenderer::prepareInstance(const std::shared_ptr<TriangleMesh>& entity)
{
    GLenum first_free_texture_binding = SceneObjectsRenderer::prepareInstance(entity);

    glActiveTexture(first_free_texture_binding + 0);
    entity->getMaterial()->bindNormalMap();

    return first_free_texture_binding + 1;
}
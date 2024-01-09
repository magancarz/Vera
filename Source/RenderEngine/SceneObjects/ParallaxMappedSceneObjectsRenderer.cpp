#include "ParallaxMappedSceneObjectsRenderer.h"

#include "GL/glew.h"

#include "Materials/Material.h"

void ParallaxMappedSceneObjectsRenderer::createSceneObjectShader()
{
    scene_object_shader = std::make_unique<ParallaxMappedSceneObjectsShader>();
    scene_object_shader->getAllUniformLocations();
    scene_object_shader->connectTextureUnits();

    parallax_mapped_scene_object_shader = dynamic_cast<ParallaxMappedSceneObjectsShader*>(scene_object_shader.get());
}

bool ParallaxMappedSceneObjectsRenderer::apply(const std::shared_ptr<TriangleMesh>& entity)
{
    return entity->isParallaxMapped();
}

GLenum ParallaxMappedSceneObjectsRenderer::prepareInstance(const std::shared_ptr<TriangleMesh>& entity)
{
    GLenum first_free_texture_binding = SceneObjectsRenderer::prepareInstance(entity);

    glActiveTexture(first_free_texture_binding + 0);
    entity->getMaterial()->bindNormalMap();

    glActiveTexture(first_free_texture_binding + 1);
    entity->getMaterial()->bindDepthMap();

    parallax_mapped_scene_object_shader->loadHeightScale(0.01f);

    return first_free_texture_binding + 2;
}
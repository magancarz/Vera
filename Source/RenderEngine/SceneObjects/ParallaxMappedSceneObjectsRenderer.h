#pragma once

#include "RenderEngine/SceneObjects/ParallaxMappedSceneObjectsShader.h"
#include "RenderEngine/SceneObjects/SceneObjectsRenderer.h"

class ParallaxMappedSceneObjectsRenderer : public SceneObjectsRenderer
{
public:
    bool apply(const std::shared_ptr<TriangleMesh>& entity) override;
    GLenum prepareInstance(const std::shared_ptr<TriangleMesh>& entity) override;

private:
    void createSceneObjectShader() override;

    ParallaxMappedSceneObjectsShader* parallax_mapped_scene_object_shader;
};

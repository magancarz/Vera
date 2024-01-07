#pragma once

#include <map>
#include <vector>

#include "RenderEngine/Shadows/ShadowMapRenderer.h"
#include "NormalMappedSceneObjectsShader.h"
#include "SceneObjectsRenderer.h"

class TriangleMesh;
class Light;

class NormalMappedSceneObjectsRenderer : public SceneObjectsRenderer
{
public:
    bool apply(const std::shared_ptr<TriangleMesh>& entity) override;
    GLenum prepareInstance(const std::shared_ptr<TriangleMesh>& entity) override;

private:
    void createSceneObjectShader() override;
};

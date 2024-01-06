#pragma once

#include <vector>
#include <memory>

#include "SceneObjectsRenderer.h"
#include "RenderEngine/GLObjects/UniformBuffer.h"

class SceneObjectsMasterRenderer
{
public:
    SceneObjectsMasterRenderer();

    void render(
            const std::map<std::shared_ptr<RawModel>, std::vector<std::weak_ptr<TriangleMesh>>>& entity_map,
            const std::vector<std::weak_ptr<Light>>& lights,
            const std::shared_ptr<Camera>& camera);

private:
    void prepareSceneObjectsRenderers();

    OutlineShader outline_shader;
    std::vector<std::unique_ptr<SceneObjectsRenderer>> scene_objects_renderers;
};
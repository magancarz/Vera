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
    void prepareLights(const std::vector<std::weak_ptr<Light>>& lights);
    void bindShadowMaps(const std::vector<std::weak_ptr<Light>>& lights);
    void prepareTransformationMatrices(const std::shared_ptr<Camera>& camera) const;
    void loadTransformationMatrix(const glm::mat4& model) const;

    UniformBuffer<LightInfo> light_info_uniform_buffer{"LightInfos"};
    UniformBuffer<TransformationMatrices> transformation_matrices_uniform_buffer{"TransformationMatrices"};

    ShadowMapRenderer shadow_map_renderer;
    std::vector<std::shared_ptr<SceneObjectsRenderer>> scene_objects_renderers =
    {
        std::make_shared<SceneObjectsRenderer>()
    };
    OutlineShader outline_shader;
};
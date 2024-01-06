#include "SceneObjectsMasterRenderer.h"

SceneObjectsMasterRenderer::SceneObjectsMasterRenderer()
{
    prepareSceneObjectsRenderers();

    outline_shader.getAllUniformLocations();
}

void SceneObjectsMasterRenderer::prepareSceneObjectsRenderers()
{
    for (const auto& scene_object_renderer : scene_objects_renderers)
    {
        scene_object_renderer->prepare();
    }
}

void SceneObjectsMasterRenderer::render(
        const std::map<std::shared_ptr<RawModel>, std::vector<std::weak_ptr<TriangleMesh>>>& entity_map,
        const std::vector<std::weak_ptr<Light>>& lights, const std::shared_ptr<Camera>& camera)
{
    for (const auto& scene_object_renderer : scene_objects_renderers)
    {
        scene_object_renderer->prepare();
    }

    outline_shader.start();
    outline_shader.loadViewAndProjectionMatrices(camera);
    ShaderProgram::stop();
}
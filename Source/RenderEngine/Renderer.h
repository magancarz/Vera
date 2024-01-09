#pragma once

#include "RenderEngine/SceneObjects/SceneObjectsRenderer.h"
#include "RenderEngine/SceneObjects/NormalMappedSceneObjectsRenderer.h"
#include "Models/AssetManager.h"
#include "Images/RayTracedImageShader.h"
#include "Camera.h"
#include "Objects/Lights/Light.h"
#include "RenderEngine/Skybox/SkyboxRenderer.h"
#include "RenderEngine/SceneObjects/ParallaxMappedSceneObjectsRenderer.h"
#include "RenderEngine/SceneObjects/SceneObjectsMasterRenderer.h"

class Renderer
{
public:
    Renderer();

    void renderScene(const std::shared_ptr<Camera>& camera, const std::vector<std::weak_ptr<Light>>& lights, const std::vector<std::weak_ptr<TriangleMesh>>& entities);
    void renderRayTracedImage(unsigned texture_id) const;

private:
    static void prepare();
    void processEntities(const std::vector<std::weak_ptr<TriangleMesh>>& entities);
    void processEntity(std::map<std::shared_ptr<RawModel>, std::vector<std::weak_ptr<TriangleMesh>>>& map, const std::weak_ptr<TriangleMesh>& entity);
    void cleanUpObjectsMaps();

    SkyboxRenderer skybox_renderer;
    SceneObjectsMasterRenderer scene_objects_master_renderer;
    RayTracedImageShader ray_traced_image_shader;
    std::map<std::shared_ptr<RawModel>, std::vector<std::weak_ptr<TriangleMesh>>> objects_map;

    RawModelAttributes quad;
    inline static const std::vector<float> quad_positions =
    {
        -1.0f, 1.0f,
        -1.0f, -1.0f,
        1.0f, 1.0f,
        1.0f, -1.0f
    };
    inline static const std::vector<float> quad_textures =
    {
        0.0f, 0.0f,
        0.0, 1.0,
        1.0, 0.0,
        1.0, 1.0,
    };
};

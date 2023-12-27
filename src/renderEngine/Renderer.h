#pragma once

#include "RenderEngine/EntityRenderer.h"
#include "../models/AssetManager.h"
#include "Shaders/RayTracedImageShader.h"
#include "Camera.h"

class Renderer
{
public:
    Renderer();

    void renderScene(const std::shared_ptr<Camera>& camera, const std::vector<std::shared_ptr<Light>>& lights, const std::vector<std::shared_ptr<TriangleMesh>>& entities);
    void renderRayTracedImage(unsigned int texture_id) const;

private:
    static void prepare();
    void processEntities(const std::vector<std::shared_ptr<TriangleMesh>>& entities);
    void processEntity(const std::shared_ptr<TriangleMesh>& entity);
    void cleanUpObjectsMaps();

    std::unique_ptr<EntityRenderer> entity_renderer;
    RayTracedImageShader ray_traced_image_shader;
    std::map<std::shared_ptr<RawModel>, std::vector<std::shared_ptr<TriangleMesh>>> entities_map;

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

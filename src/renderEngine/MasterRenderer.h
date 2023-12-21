#pragma once

#include "RenderEngine/EntityRenderer.h"
#include "../models/AssetManager.h"
#include "Shaders/RayTracedImageShader.h"
#include "Objects/Camera.h"

class MasterRenderer
{
public:
    MasterRenderer();

    void render(const std::shared_ptr<Camera>& camera) const;
    void renderRayTracedImage(unsigned int texture_id) const;

    static void prepare();

    void processEntity(const std::shared_ptr<Object>& entity);
    void processEntities(const std::vector<std::shared_ptr<Object>>& entities);

    void cleanUpObjectsMaps();

private:
    std::unique_ptr<EntityRenderer> entity_renderer;
    RayTracedImageShader ray_traced_image_shader;
    std::map<std::shared_ptr<RawModel>, std::vector<std::shared_ptr<Object>>> entities_map;

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

#pragma once

#include "RenderEngine/SceneObjects/SceneObjectsRenderer.h"
#include "RenderEngine/SceneObjects/NormalMappedSceneObjectsRenderer.h"
#include "Models/AssetManager.h"
#include "Images/RayTracedImageShader.h"
#include "Camera.h"
#include "Objects/Lights/Light.h"
#include "RenderEngine/Skybox/SkyboxRenderer.h"
#include "RenderEngine/SceneObjects/ParallaxMappedSceneObjectsRenderer.h"
#include "RenderEngine/SceneObjects/DeferredShadingRenderer.h"
#include "RenderEngine/HDR/HDRShader.h"

class Renderer
{
public:
    Renderer();

    void renderScene(const std::shared_ptr<Camera>& camera, const std::vector<std::weak_ptr<Light>>& lights, const std::vector<std::weak_ptr<TriangleMesh>>& entities);
    void renderRayTracedImage(unsigned texture_id) const;

private:
    static void prepare();
    void createHDRFramebuffer();
    void processEntities(const std::vector<std::weak_ptr<TriangleMesh>>& entities);
    void processEntity(std::map<std::shared_ptr<RawModel>, std::vector<std::weak_ptr<TriangleMesh>>>& map, const std::weak_ptr<TriangleMesh>& entity);
    void applyToneMappingAndRender();
    void cleanUpObjectsMaps();

    utils::FBO hdr_fbo;
    utils::Texture hdr_color_texture;
    utils::Renderbuffer hdr_rbo_depth;
    HDRShader hdr_shader;

    SkyboxRenderer skybox_renderer;
    DeferredShadingRenderer deferred_shading_renderer;
    RayTracedImageShader ray_traced_image_shader;
    std::map<std::shared_ptr<RawModel>, std::vector<std::weak_ptr<TriangleMesh>>> objects_map;
};

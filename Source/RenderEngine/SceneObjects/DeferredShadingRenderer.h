#pragma once

#include <vector>
#include <memory>

#include "ParallaxMappedSceneObjectsRenderer.h"
#include "NormalMappedSceneObjectsRenderer.h"
#include "RenderEngine/GLObjects/UniformBuffer.h"
#include "LightingPassRenderer.h"
#include "LightObjectsShader.h"
#include "OutlineMarkShader.h"

class DeferredShadingRenderer
{
public:
    DeferredShadingRenderer();

    void renderSceneObjects(
            const std::map<std::shared_ptr<RawModel>, std::vector<std::weak_ptr<TriangleMesh>>>& entity_map,
            const std::vector<std::weak_ptr<Light>>& lights,
            const std::shared_ptr<Camera>& camera);

private:
    void createGBuffer();
    void prepareSceneObjectsRenderers();
    void prepareLights(const std::vector<std::weak_ptr<Light>>& lights);
    void bindShadowMaps(const std::vector<std::weak_ptr<Light>>& lights);
    void prepareTransformationMatrices(const std::shared_ptr<Camera>& camera) const;
    void loadTransformationMatrix(const glm::mat4& model) const;
    void renderShadowMap(
        const std::map<std::shared_ptr<RawModel>, std::vector<std::weak_ptr<TriangleMesh>>>& entity_map,
        const std::vector<std::weak_ptr<Light>>& lights);
    void renderLightObjects(const std::map<std::shared_ptr<RawModel>, std::vector<std::shared_ptr<TriangleMesh>>>& light_objects);
    void renderOutlines();

    utils::FBO g_buffer;
    utils::Texture g_position;
    utils::Texture g_normal;
    utils::Texture g_color_spec;
    utils::Renderbuffer rbo_depth;
    utils::Renderbuffer rbo_stencil;

    UniformBuffer<LightInfo> light_info_uniform_buffer{"LightInfos"};
    UniformBuffer<TransformationMatrices> transformation_matrices_uniform_buffer{"TransformationMatrices"};

    ShadowMapRenderer shadow_map_renderer;
    std::vector<std::shared_ptr<SceneObjectsRenderer>> scene_objects_renderers =
    {
        std::make_shared<ParallaxMappedSceneObjectsRenderer>(),
        std::make_shared<NormalMappedSceneObjectsRenderer>(),
        std::make_shared<SceneObjectsRenderer>()
    };
    LightingPassRenderer lighting_pass_renderer;
    OutlineMarkShader outline_mark_shader;
    OutlineShader outline_shader;
    LightObjectsShader light_objects_shader;
};
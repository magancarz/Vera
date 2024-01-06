#pragma once

#include <map>
#include <vector>

#include "OutlineShader.h"
#include "SceneObjectsShader.h"
#include "Objects/Object.h"
#include "Models/AssetManager.h"
#include "RenderEngine/Shadows/ShadowMapRenderer.h"
#include "NormalMappedSceneObjectsShader.h"
#include "ParallaxMappedSceneObjectsShader.h"

class TriangleMesh;
class Light;

class ParallaxMappedSceneObjectsRenderer
{
public:
    ParallaxMappedSceneObjectsRenderer();

    void render(
        const std::map<std::shared_ptr<RawModel>, std::vector<std::weak_ptr<TriangleMesh>>>& entity_map,
        const std::vector<std::weak_ptr<Light>>& lights,
        const std::shared_ptr<Camera>& camera);

private:
    void prepareTexturedModel(const std::shared_ptr<RawModel>& raw_model) const;
    void prepareInstance(const std::weak_ptr<TriangleMesh>& entity);

    static void unbindTexturedModel();

    ParallaxMappedSceneObjectsShader parallax_mapped_scene_objects_shader;
    OutlineShader outline_shader;

    GLenum current_texture{GL_TEXTURE0};
};

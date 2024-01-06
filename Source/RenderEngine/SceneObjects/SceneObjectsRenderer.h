#pragma once

#include <map>
#include <vector>

#include "OutlineShader.h"
#include "SceneObjectsShader.h"
#include "Objects/Object.h"
#include "Models/AssetManager.h"
#include "RenderEngine/Shadows/ShadowMapRenderer.h"

class TriangleMesh;
class Light;

class SceneObjectsRenderer
{
public:
    virtual bool apply();
    void render(
        const std::map<std::shared_ptr<RawModel>, std::vector<std::weak_ptr<TriangleMesh>>>& entity_map,
        const std::vector<std::weak_ptr<Light>>& lights,
        const std::shared_ptr<Camera>& camera);

    virtual void prepare();

private:
    virtual void createSceneObjectShader();
    virtual void prepareTexturedModel(const std::shared_ptr<RawModel>& raw_model) const;
    virtual void unbindTexturedModel();
    virtual void prepareInstance(const std::weak_ptr<TriangleMesh>& entity);

    std::unique_ptr<SceneObjectsShader> static_shader;

    GLenum current_texture{GL_TEXTURE0};
};

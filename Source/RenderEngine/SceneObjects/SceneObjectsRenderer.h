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
    virtual void prepare();

    virtual bool apply();
    virtual void prepareTexturedModel(const std::shared_ptr<RawModel>& raw_model) const;
    virtual void prepareInstance(const std::weak_ptr<TriangleMesh>& entity);
    void prepareShader();
    virtual void unbindTexturedModel();

    template <typename T>
    void bindUniformBuffer(const UniformBuffer<T>& uniform_buffer)
    {
        scene_object_shader->bindUniformBlockToShader(uniform_buffer.getName(), uniform_buffer.getUniformBlockIndex());
    }

private:
    virtual void createSceneObjectShader();

    std::unique_ptr<SceneObjectsShader> scene_object_shader;
};

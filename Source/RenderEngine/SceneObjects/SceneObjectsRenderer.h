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

    virtual bool apply(const std::shared_ptr<TriangleMesh>& entity);
    virtual void prepareShader();
    virtual GLenum prepareInstance(const std::shared_ptr<TriangleMesh>& entity);

    template <typename T>
    void bindUniformBuffer(const UniformBuffer<T>& uniform_buffer)
    {
        scene_object_shader->bindUniformBlockToShader(uniform_buffer.getName(), uniform_buffer.getUniformBlockIndex());
    }

protected:
    std::unique_ptr<SceneObjectsShader> scene_object_shader;

private:
    virtual void createSceneObjectShader();
};

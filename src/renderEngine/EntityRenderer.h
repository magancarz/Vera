#pragma once

#include <map>
#include <vector>

#include "Shaders/OutlineShader.h"
#include "Shaders/StaticShader.h"
#include "Objects/Object.h"
#include "../models/AssetManager.h"

class TriangleMesh;
class Light;

class EntityRenderer
{
public:
    EntityRenderer();

    void render(
        const std::map<std::shared_ptr<RawModel>, std::vector<std::weak_ptr<TriangleMesh>>>& entity_map,
        const std::vector<std::weak_ptr<Light>>& lights,
        const std::shared_ptr<Camera>& camera) const;

private:
    void prepareTexturedModel(const std::shared_ptr<RawModel>& raw_model) const;
    void prepareInstance(const std::weak_ptr<TriangleMesh>& entity) const;

    static void unbindTexturedModel();

    StaticShader static_shader;
    OutlineShader outline_shader;
};

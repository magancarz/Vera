#pragma once

#include <memory>
#include <map>
#include <vector>

#include "ShadowMapShader.h"
#include "Models/RawModel.h"
#include "Objects/TriangleMesh.h"
#include "Objects/Lights/Light.h"

class ShadowMapRenderer
{
public:
    ShadowMapRenderer();

    void renderSceneToDepthBuffers(
        const std::map<std::shared_ptr<RawModel>, std::vector<std::weak_ptr<TriangleMesh>>>& entity_map,
        const std::vector<std::weak_ptr<Light>>& lights);

private:
    void createShadowMapFrameBuffer();

    void draw(
        const std::map<std::shared_ptr<RawModel>, std::vector<std::weak_ptr<TriangleMesh>>>& entity_map,
        const std::weak_ptr<Light>& light);
    void prepareTexturedModel(const std::shared_ptr<RawModel>& raw_model) const;
    void unbindTexturedModel();
    void prepareInstance(const std::weak_ptr<TriangleMesh>& entity, const std::weak_ptr<Light>& light) const;

    utils::FBO depth_map_FBO;
};
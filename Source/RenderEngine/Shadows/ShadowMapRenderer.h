#pragma once

#include <memory>
#include <map>
#include <vector>

#include "ShadowMapShader.h"
#include "Models/RawModel.h"
#include "Objects/TriangleMesh.h"

class ShadowMapRenderer
{
public:
    ShadowMapRenderer();

    void renderSceneToDepthBuffer(
        const std::map<std::shared_ptr<RawModel>, std::vector<std::weak_ptr<TriangleMesh>>>& entity_map);

    void bindShadowMapTexture() const;
    glm::mat4 getToLightSpaceTransform() const;

private:
    void createDepthMapTexture();
    void createShadowMapFrameBuffer();
    void configureShaderAndMatrices();

    void draw(const std::map<std::shared_ptr<RawModel>, std::vector<std::weak_ptr<TriangleMesh>>>& entity_map);
    void prepareTexturedModel(const std::shared_ptr<RawModel>& raw_model) const;
    void unbindTexturedModel();
    void prepareInstance(const std::weak_ptr<TriangleMesh>& entity) const;

    ShadowMapShader shadow_map_shader;

    unsigned int shadow_map_width = 1024;
    unsigned int shadow_map_height = 1024;
    float shadow_map_projection_x_span = 10.f;
    float shadow_map_projection_y_span = 10.f;
    float near_plane = 1.0f;
    float far_plane = 30.f;

    unsigned int depth_map_FBO;
    unsigned int depth_map_texture;

    glm::mat4 to_light_space_transform;
    glm::mat4 light_projection;
    glm::mat4 light_view;
};
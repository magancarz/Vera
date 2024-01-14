#pragma once

#include "SkyboxShader.h"
#include "Models/RawModel.h"
#include "RenderEngine/Camera.h"

class SkyboxRenderer
{
public:
    SkyboxRenderer();

    void renderSkybox(
        const utils::FBO& hdr_fbo,
        const std::shared_ptr<Camera>& camera,
        const std::shared_ptr<utils::Texture>& skybox);

    void renderSkybox(
        const utils::FBO& hdr_fbo,
        const std::shared_ptr<Camera>& camera);

private:
    SkyboxShader skybox_shader;
    std::shared_ptr<RawModel> cube_model;
    std::shared_ptr<utils::Texture> cube_map;
};

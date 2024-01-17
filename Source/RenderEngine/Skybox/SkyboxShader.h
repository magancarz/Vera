#pragma once

#include <memory>

#include "Shaders/ShaderProgram.h"

class Camera;

class SkyboxShader : public ShaderProgram
{
public:
    SkyboxShader();

    void loadViewMatrix(const std::shared_ptr<Camera>& camera);
    void loadProjectionMatrix(const std::shared_ptr<Camera>& camera);

    void connectTextureUnits();
    void getAllUniformLocations() override;

private:
    int location_view_matrix;
    int location_projection_matrix;
    int location_skybox;
};

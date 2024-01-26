#include "SkyboxShader.h"

#include "RenderEngine/Camera.h"

SkyboxShader::SkyboxShader()
    : ShaderProgram(
        "Source/RenderEngine/Skybox/skybox_vert.glsl",
        "Source/RenderEngine/Skybox/skybox_default_frag.glsl") {}

void SkyboxShader::loadViewMatrix(const std::shared_ptr<Camera>& camera)
{
    glm::mat4 view = glm::mat4(glm::mat3(camera->getCameraViewMatrix()));
    loadMatrix(location_view_matrix, view);
}

void SkyboxShader::loadProjectionMatrix(const std::shared_ptr<Camera>& camera)
{
    loadMatrix(location_projection_matrix, camera->getPerspectiveProjectionMatrix());
}

void SkyboxShader::connectTextureUnits()
{
    loadInt(location_skybox, 0);
}

void SkyboxShader::getAllUniformLocations()
{
    location_view_matrix = getUniformLocation("view");
    location_projection_matrix = getUniformLocation("projection");
    location_skybox = getUniformLocation("skybox");
}
#include "SceneObjectsShader.h"

#include <memory>
#include <ranges>

#include "Materials/Material.h"
#include "Objects/Object.h"
#include "RenderEngine/Camera.h"
#include "Objects/Lights/Light.h"
#include "Objects/Lights/PointLight.h"

SceneObjectsShader::SceneObjectsShader()
    : ShaderProgram("Resources/Shaders/SceneObjectVert.glsl", "Resources/Shaders/SceneObjectFrag.glsl"),
      light_info_uniform_buffer(LIGHT_INFO_UNIFORM_BLOCK_NAME),
      projection_matrices_uniform_buffer(PROJECTION_MATRICES_UNIFORM_BLOCK_NAME)
{
    prepareUniformBuffers();
    connectTextureUnits();
}

void SceneObjectsShader::prepareUniformBuffers()
{
    bindUniformBlockToShader(light_info_uniform_buffer.getName(), light_info_uniform_buffer.getUniformBlockIndex());
    bindUniformBlockToShader(projection_matrices_uniform_buffer.getName(), projection_matrices_uniform_buffer.getUniformBlockIndex());
}

SceneObjectsShader::SceneObjectsShader(const std::string& vertex_file, const std::string& fragment_file)
    : ShaderProgram(vertex_file, fragment_file), light_info_uniform_buffer(LIGHT_INFO_UNIFORM_BLOCK_NAME),
      projection_matrices_uniform_buffer(PROJECTION_MATRICES_UNIFORM_BLOCK_NAME)
{
    prepareUniformBuffers();
    connectTextureUnits();
}

SceneObjectsShader::SceneObjectsShader(
        const std::string& vertex_file,
        const std::string& geometry_file,
        const std::string& fragment_file)
    : ShaderProgram(vertex_file, geometry_file, fragment_file), light_info_uniform_buffer(LIGHT_INFO_UNIFORM_BLOCK_NAME),
      projection_matrices_uniform_buffer(PROJECTION_MATRICES_UNIFORM_BLOCK_NAME)
{
    prepareUniformBuffers();
    connectTextureUnits();
}

void SceneObjectsShader::loadTransformationMatrix(const glm::mat4& matrix) const
{
    loadMatrix(location_transformation_matrix, matrix);
}

void SceneObjectsShader::loadViewAndProjectionMatrices(const std::shared_ptr<Camera>& camera) const
{
    const auto view = camera->getCameraViewMatrix();
    const auto projection = camera->getPerspectiveProjectionMatrix();
    projection_matrices_uniform_buffer.setValue({view, projection});
}

void SceneObjectsShader::loadReflectivity(float reflectivity) const
{
    loadFloat(location_reflectivity, reflectivity);
}

void SceneObjectsShader::connectTextureUnits() const
{
    loadInt(location_model_texture, 0);
}

void SceneObjectsShader::getAllUniformLocations()
{
    location_transformation_matrix = getUniformLocation("model");
    location_model_texture = getUniformLocation("color_texture_sampler");
    location_reflectivity = getUniformLocation("reflectivity");
    location_shadow_map = getUniformLocation("shadow_map");
}

void SceneObjectsShader::loadLights(const std::vector<std::weak_ptr<Light>>& lights) const
{
    glActiveTexture(GL_TEXTURE1);
    lights[0].lock()->getShadowMap()->bindTexture();
    loadInt(location_shadow_map, 1);

    for (const auto& light : lights)
    {
        LightInfo light_info
        {
            light.lock()->getPosition(),
            light.lock()->getLightColor(),
            light.lock()->getAttenuation()
        };
        light_info_uniform_buffer.setValue(light_info);
    }
}

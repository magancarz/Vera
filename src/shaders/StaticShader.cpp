#include "StaticShader.h"

#include <memory>
#include <ranges>

#include "Materials/Material.h"
#include "Objects/Object.h"
#include "renderEngine/Camera.h"
#include "Objects/Lights/Light.h"

StaticShader::StaticShader()
    : ShaderProgram("res/shaders/vert.glsl", "res/shaders/frag.glsl") {}

void StaticShader::loadTransformationMatrix(const glm::mat4& matrix) const
{
    loadMatrix(location_transformation_matrix, matrix);
}

void StaticShader::loadProjectionMatrix(const std::shared_ptr<Camera>& camera) const
{
    const auto projection_matrix = camera->getPerspectiveProjectionMatrix();
    loadMatrix(location_projection_matrix, projection_matrix);
}

void StaticShader::loadViewMatrix(const std::shared_ptr<Camera>& camera) const
{
    const auto view = camera->getCameraViewMatrix();
    loadMatrix(location_view_matrix, view);
}

void StaticShader::loadReflectivity(float reflectivity) const
{
    loadFloat(location_reflectivity, reflectivity);
}

void StaticShader::connectTextureUnits() const
{
    loadInt(location_model_texture, 0);
}

void StaticShader::bindAttributes()
{
    bindAttribute(0, "position");
    bindAttribute(1, "texture_coords");
    bindAttribute(2, "normal");
}

void StaticShader::getAllUniformLocations()
{
    location_transformation_matrix = getUniformLocation("model");
    location_projection_matrix = getUniformLocation("proj");
    location_view_matrix = getUniformLocation("view");

    location_model_texture = getUniformLocation("texture_sampler");

    for (const int i : std::views::iota(0, MAX_LIGHTS))
    {
        location_light_position[i] = getUniformLocation("lights[" + std::to_string(i) + "].light_position");
        location_light_direction[i] = getUniformLocation("lights[" + std::to_string(i) + "].light_direction");
        location_light_color[i] = getUniformLocation("lights[" + std::to_string(i) + "].light_color");
        location_attenuation[i] = getUniformLocation("lights[" + std::to_string(i) + "].attenuation");
        location_cutoff_angle[i] = getUniformLocation("lights[" + std::to_string(i) + "].cutoff_angle");
        location_cutoff_angle_offset[i] = getUniformLocation("lights[" + std::to_string(i) + "].cutoff_angle_offset");
    }

    location_reflectivity = getUniformLocation("reflectivity");
}

void StaticShader::loadLights(const std::vector<std::weak_ptr<Light>>& lights) const
{
    size_t loaded_lights = 0;
    for (const auto& light : lights)
    {
        loadVector3(location_light_position[loaded_lights], light.lock()->getPosition());
        loadVector4(location_light_direction[loaded_lights], light.lock()->getLightDirection());
        loadVector3(location_light_color[loaded_lights], light.lock()->getLightColor());
        loadVector3(location_attenuation[loaded_lights], light.lock()->getAttenuation());
        loadFloat(location_cutoff_angle[loaded_lights], light.lock()->getCutoffAngle());
        loadFloat(location_cutoff_angle_offset[loaded_lights], light.lock()->getCutoffAngleOffset());

        if (++loaded_lights >= MAX_LIGHTS)
        {
            std::cerr << "Number of lights in the scene is larger than capable value: " << MAX_LIGHTS << ".\n";
            return;
        }
    }

    for (size_t i = loaded_lights; i < MAX_LIGHTS; ++i)
    {
        loadVector3(location_light_position[loaded_lights], glm::vec3(0));
        loadVector4(location_light_direction[loaded_lights], glm::vec4{0, 0, 0, 1});
        loadVector3(location_light_color[loaded_lights], glm::vec3(0));
        loadVector3(location_attenuation[loaded_lights], {1, 0, 0});
        loadFloat(location_cutoff_angle[loaded_lights], 0);
        loadFloat(location_cutoff_angle_offset[loaded_lights], 0);
    }
}

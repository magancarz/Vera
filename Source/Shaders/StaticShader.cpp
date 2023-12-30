#include "StaticShader.h"

#include <memory>
#include <ranges>

#include "Materials/Material.h"
#include "Objects/Object.h"
#include "renderEngine/Camera.h"
#include "Objects/Lights/Light.h"

StaticShader::StaticShader()
    : ShaderProgram("Resources/Shaders/vert.glsl", "Resources/Shaders/frag.glsl") {}

void StaticShader::loadTransformationMatrix(const glm::mat4& matrix) const
{
    loadMatrix(location_transformation_matrix, matrix);
}

void StaticShader::loadViewMatrix(const std::shared_ptr<Camera>& camera) const
{
    const auto view = camera->getCameraViewMatrix();
    loadMatrix(location_view_matrix, view);
}

void StaticShader::loadProjectionMatrix(const std::shared_ptr<Camera>& camera) const
{
    const auto projection_matrix = camera->getPerspectiveProjectionMatrix();
    loadMatrix(location_projection_matrix, projection_matrix);
}

void StaticShader::loadLightViewMatrix(const glm::mat4& matrix) const
{
    loadMatrix(location_light_view_matrix, matrix);
}

void StaticShader::loadReflectivity(float reflectivity) const
{
    loadFloat(location_reflectivity, reflectivity);
}

void StaticShader::loadHeightScale(float height_scale) const
{
    loadFloat(location_height_scale, height_scale);
}

void StaticShader::loadNormalMapLoadedBool(bool value) const
{
    loadInt(location_normal_map_loaded, value);
}

void StaticShader::loadDepthMapLoadedBool(bool value) const
{
    loadInt(location_depth_map_loaded, value);
}

void StaticShader::connectTextureUnits() const
{
    loadInt(location_model_texture, 0);
    loadInt(location_normal_texture, 1);
    loadInt(location_depth_texture, 2);
    loadInt(location_shadow_map_texture, 3);
}

void StaticShader::getAllUniformLocations()
{
    location_transformation_matrix = getUniformLocation("model");
    location_view_matrix = getUniformLocation("view");
    location_projection_matrix = getUniformLocation("proj");
    location_light_view_matrix = getUniformLocation("light_view");

    location_model_texture = getUniformLocation("color_texture_sampler");
    location_normal_texture = getUniformLocation("normal_texture_sampler");
    location_depth_texture = getUniformLocation("depth_texture_sampler");
    location_shadow_map_texture = getUniformLocation("shadow_map_texture_sampler");

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
    location_height_scale = getUniformLocation("height_scale");

    location_normal_map_loaded = getUniformLocation("normal_map_loaded");
    location_depth_map_loaded = getUniformLocation("depth_map_loaded");
}

void StaticShader::loadLights(const std::vector<std::weak_ptr<Light>>& lights) const
{
    size_t loaded_lights = 0;
    for (const auto& light : lights)
    {
        loadVector3(location_light_position[loaded_lights], light.lock()->getPosition());
        loadVector3(location_light_direction[loaded_lights], light.lock()->getLightDirection());
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
        loadVector3(location_light_direction[loaded_lights], glm::vec3{0, 0, 0});
        loadVector3(location_light_color[loaded_lights], glm::vec3(0));
        loadVector3(location_attenuation[loaded_lights], {1, 0, 0});
        loadFloat(location_cutoff_angle[loaded_lights], 0);
        loadFloat(location_cutoff_angle_offset[loaded_lights], 0);
    }
}

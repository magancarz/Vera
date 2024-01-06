#include "ParallaxMappedSceneObjectsShader.h"

#include <memory>
#include <ranges>

#include "Materials/Material.h"
#include "RenderEngine/Camera.h"
#include "Objects/Lights/Light.h"

ParallaxMappedSceneObjectsShader::ParallaxMappedSceneObjectsShader()
    : ShaderProgram("Resources/Shaders/ParallaxMappedSceneObjectVert.glsl", "Resources/Shaders/ParallaxMappedSceneObjectFrag.glsl") {}

void ParallaxMappedSceneObjectsShader::loadTransformationMatrix(const glm::mat4& matrix) const
{
    loadMatrix(location_transformation_matrix, matrix);
}

void ParallaxMappedSceneObjectsShader::loadViewMatrix(const std::shared_ptr<Camera>& camera) const
{
    const auto view = camera->getCameraViewMatrix();
    loadMatrix(location_view_matrix, view);
}

void ParallaxMappedSceneObjectsShader::loadProjectionMatrix(const std::shared_ptr<Camera>& camera) const
{
    const auto projection_matrix = camera->getPerspectiveProjectionMatrix();
    loadMatrix(location_projection_matrix, projection_matrix);
}

void ParallaxMappedSceneObjectsShader::loadLightsCount(size_t count)
{
    loadInt(location_lights_count, count);
}

void ParallaxMappedSceneObjectsShader::loadReflectivity(float reflectivity) const
{
    loadFloat(location_reflectivity, reflectivity);
}

void ParallaxMappedSceneObjectsShader::loadHeightScale(float height_scale) const
{
    loadFloat(location_height_scale, height_scale);
}

void ParallaxMappedSceneObjectsShader::connectTextureUnits(GLenum texture) const
{
    int texture_index = texture - GL_TEXTURE0;
    loadInt(location_model_texture, texture_index + 0);
    loadInt(location_normal_texture, texture_index + 1);
    loadInt(location_depth_texture, texture_index + 2);
}

void ParallaxMappedSceneObjectsShader::loadTextureIndexToCubeShadowMap(size_t cube_shadow_map_index, unsigned int texture_index)
{
    loadInt(location_shadow_cube_map_textures[cube_shadow_map_index], texture_index);
}

void ParallaxMappedSceneObjectsShader::getAllUniformLocations()
{
    location_transformation_matrix = getUniformLocation("model");
    location_view_matrix = getUniformLocation("view");
    location_projection_matrix = getUniformLocation("proj");

    location_model_texture = getUniformLocation("color_texture_sampler");
    location_normal_texture = getUniformLocation("normal_texture_sampler");
    location_depth_texture = getUniformLocation("depth_texture_sampler");

    location_lights_count = getUniformLocation("lights_count");
    for (const int i : std::views::iota(0, MAX_LIGHTS))
    {
        location_shadow_cube_map_textures[i] = getUniformLocation("shadow_cube_map_texture_sampler[" + std::to_string(i) + "]");
        location_light_position[i] = getUniformLocation("lights[" + std::to_string(i) + "].light_position");
        location_light_color[i] = getUniformLocation("lights[" + std::to_string(i) + "].light_color");
        location_attenuation[i] = getUniformLocation("lights[" + std::to_string(i) + "].attenuation");
    }

    location_reflectivity = getUniformLocation("reflectivity");
    location_height_scale = getUniformLocation("height_scale");
}

void ParallaxMappedSceneObjectsShader::loadLights(const std::vector<std::weak_ptr<Light>>& lights) const
{
    size_t loaded_lights = 0;
    for (const auto& light : lights)
    {
        loadVector3(location_light_position[loaded_lights], light.lock()->getPosition());
        loadVector3(location_light_color[loaded_lights], light.lock()->getLightColor());
        loadVector3(location_attenuation[loaded_lights], light.lock()->getAttenuation());

        if (++loaded_lights >= MAX_LIGHTS)
        {
            std::cerr << "Number of lights in the scene is larger than capable value: " << MAX_LIGHTS << ".\n";
            return;
        }
    }

    for (size_t i = loaded_lights; i < MAX_LIGHTS; ++i)
    {
        loadVector3(location_light_position[loaded_lights], glm::vec3(0));
        loadVector3(location_light_color[loaded_lights], glm::vec3(0));
        loadVector3(location_attenuation[loaded_lights], {1, 0, 0});
    }
}

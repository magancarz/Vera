#include "StaticShader.h"

#include <memory>
#include <ranges>

#include "Materials/Material.h"
#include "Objects/Object.h"
#include "Objects/Camera.h"

StaticShader::StaticShader()
    : ShaderProgram("res/shaders/vert.glsl", "res/shaders/frag.glsl") {}

void StaticShader::loadTransformationMatrix(const glm::mat4& matrix) const
{
    loadMatrix(location_transformation_matrix, matrix);
}

void StaticShader::loadProjectionMatrix(const std::shared_ptr<Camera>& camera) const
{
    const auto projection_matrix = camera->getProjectionMatrix();
    loadMatrix(location_projection_matrix, projection_matrix);
}

void StaticShader::loadViewMatrix(const std::shared_ptr<Camera>& camera) const
{
    const auto view = camera->getView();
    loadMatrix(location_view_matrix, view);
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
        location_light_position[i] = getUniformLocation("light_position[" + std::to_string(i) + "]");
        location_light_color[i] = getUniformLocation("light_color[" + std::to_string(i) + "]");
    }
}

void StaticShader::loadLights(const std::map<std::shared_ptr<RawModel>, std::vector<std::shared_ptr<Object>>>& entity_map) const
{
    size_t loaded_lights = 0;
    for (const auto& [model, objects] : entity_map)
    {
        for (const auto& object : objects)
        {
            if (object->isEmittingSomeLight())
            {
                loadVector3(location_light_position[loaded_lights], object->getPosition());
                loadVector3(location_light_color[loaded_lights], object->getMaterial()->getColor({0.5, 0.5}));

                if (++loaded_lights >= MAX_LIGHTS)
                {
                    std::cerr << "Number of lights in the scene is larger than capable value: " << MAX_LIGHTS << ".\n";
                    return;
                }
            }
        }
    }

    for (int i = loaded_lights; i < MAX_LIGHTS; ++i)
    {
        loadVector3(location_light_position[loaded_lights], glm::vec3(0));
        loadVector3(location_light_color[loaded_lights], glm::vec3(0));
    }
}

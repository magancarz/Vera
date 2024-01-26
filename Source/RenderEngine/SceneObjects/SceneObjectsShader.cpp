#include "SceneObjectsShader.h"

#include <ranges>

#include "Materials/Material.h"
#include "Objects/Lights/Light.h"
#include "RenderEngine/RendererDefines.h"

SceneObjectsShader::SceneObjectsShader()
    : ShaderProgram("Resources/Shaders/SceneObjectVert.glsl", "Resources/Shaders/SceneObjectFrag.glsl") {}

SceneObjectsShader::SceneObjectsShader(const std::string& vertex_file, const std::string& fragment_file)
    : ShaderProgram(vertex_file, fragment_file) {}

SceneObjectsShader::SceneObjectsShader(
        const std::string& vertex_file,
        const std::string& geometry_file,
        const std::string& fragment_file)
    : ShaderProgram(vertex_file, geometry_file, fragment_file) {}

void SceneObjectsShader::connectTextureUnits()
{
    loadInt(location_model_texture, RendererDefines::MODEL_TEXTURES_STARTING_INDEX + 0);
}

void SceneObjectsShader::getAllUniformLocations()
{
    location_reflectivity = getUniformLocation("reflectivity");
    location_model_texture = getUniformLocation("color_texture_sampler");
}

void SceneObjectsShader::loadReflectivity(float reflectivity)
{
    loadFloat(location_reflectivity, reflectivity);
}
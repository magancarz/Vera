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

size_t SceneObjectsShader::connectTextureUnits() const
{
    loadInt(location_shadow_map, RendererDefines::SHADOW_MAPS_TEXTURES_STARTING_INDEX);
    loadInt(location_model_texture, RendererDefines::MODEL_TEXTURES_STARTING_INDEX + 0);

    return RendererDefines::MODEL_TEXTURES_STARTING_INDEX + 1;
}

void SceneObjectsShader::getAllUniformLocations()
{
    location_reflectivity = getUniformLocation("reflectivity");
    location_model_texture = getUniformLocation("color_texture_sampler");
    location_shadow_map = getUniformLocation("shadow_map");
}

void SceneObjectsShader::loadReflectivity(float reflectivity) const
{
    loadFloat(location_reflectivity, reflectivity);
}
#include "ParallaxMappedSceneObjectsShader.h"

ParallaxMappedSceneObjectsShader::ParallaxMappedSceneObjectsShader()
    : SceneObjectsShader("Resources/Shaders/ParallaxMappedSceneObjectVert.glsl", "Resources/Shaders/ParallaxMappedSceneObjectFrag.glsl") {}

size_t ParallaxMappedSceneObjectsShader::connectTextureUnits() const
{
    size_t first_free_texture = SceneObjectsShader::connectTextureUnits();

    loadInt(location_normal_texture, first_free_texture + 0);
    loadInt(location_depth_texture, first_free_texture + 1);

    return first_free_texture + 2;
}

void ParallaxMappedSceneObjectsShader::getAllUniformLocations()
{
    SceneObjectsShader::getAllUniformLocations();

    location_normal_texture = getUniformLocation("normal_texture_sampler");
    location_depth_texture = getUniformLocation("depth_texture_sampler");
    location_height_scale = getUniformLocation("height_scale");
}

void ParallaxMappedSceneObjectsShader::loadHeightScale(float height_scale) const
{
    loadFloat(location_height_scale, height_scale);
}
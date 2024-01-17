#include "ParallaxMappedSceneObjectsShader.h"
#include "RenderEngine/RendererDefines.h"

ParallaxMappedSceneObjectsShader::ParallaxMappedSceneObjectsShader()
    : SceneObjectsShader("Resources/Shaders/ParallaxMappedSceneObjectVert.glsl", "Resources/Shaders/ParallaxMappedSceneObjectFrag.glsl") {}

void ParallaxMappedSceneObjectsShader::connectTextureUnits()
{
    SceneObjectsShader::connectTextureUnits();

    loadInt(location_normal_texture, RendererDefines::MODEL_TEXTURES_STARTING_INDEX + 1);
    loadInt(location_depth_texture, RendererDefines::MODEL_TEXTURES_STARTING_INDEX + 2);
}

void ParallaxMappedSceneObjectsShader::getAllUniformLocations()
{
    SceneObjectsShader::getAllUniformLocations();

    location_normal_texture = getUniformLocation("normal_texture_sampler");
    location_depth_texture = getUniformLocation("depth_texture_sampler");
    location_height_scale = getUniformLocation("height_scale");
}

void ParallaxMappedSceneObjectsShader::loadHeightScale(float height_scale)
{
    loadFloat(location_height_scale, height_scale);
}
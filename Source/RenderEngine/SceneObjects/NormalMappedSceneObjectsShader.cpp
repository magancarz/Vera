#include "NormalMappedSceneObjectsShader.h"
#include "RenderEngine/RendererDefines.h"

NormalMappedSceneObjectsShader::NormalMappedSceneObjectsShader()
    : SceneObjectsShader("Resources/Shaders/NormalMappedSceneObjectVert.glsl", "Resources/Shaders/NormalMappedSceneObjectFrag.glsl") {}

void NormalMappedSceneObjectsShader::connectTextureUnits()
{
    SceneObjectsShader::connectTextureUnits();

    loadInt(location_normal_texture, RendererDefines::MODEL_TEXTURES_STARTING_INDEX + 1);
}

void NormalMappedSceneObjectsShader::getAllUniformLocations()
{
    SceneObjectsShader::getAllUniformLocations();

    location_normal_texture = getUniformLocation("normal_texture_sampler");
}

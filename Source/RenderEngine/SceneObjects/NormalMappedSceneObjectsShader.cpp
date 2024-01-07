#include "NormalMappedSceneObjectsShader.h"

#include <memory>
#include <ranges>

#include "Materials/Material.h"
#include "RenderEngine/Camera.h"
#include "Objects/Lights/Light.h"

NormalMappedSceneObjectsShader::NormalMappedSceneObjectsShader()
    : SceneObjectsShader("Resources/Shaders/NormalMappedSceneObjectVert.glsl", "Resources/Shaders/NormalMappedSceneObjectFrag.glsl") {}

size_t NormalMappedSceneObjectsShader::connectTextureUnits() const
{
    size_t first_free_texture = SceneObjectsShader::connectTextureUnits();
    loadInt(location_normal_texture, first_free_texture + 0);
    return first_free_texture + 1;
}

void NormalMappedSceneObjectsShader::getAllUniformLocations()
{
    SceneObjectsShader::getAllUniformLocations();

    location_normal_texture = getUniformLocation("normal_texture_sampler");
}

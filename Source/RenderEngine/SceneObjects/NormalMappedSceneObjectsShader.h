#pragma once

#include "RenderEngine/SceneObjects/SceneObjectsShader.h"

class NormalMappedSceneObjectsShader : public SceneObjectsShader
{
public:
    NormalMappedSceneObjectsShader();

    size_t connectTextureUnits() const override;
    void getAllUniformLocations() override;

private:
    int location_normal_texture;
};

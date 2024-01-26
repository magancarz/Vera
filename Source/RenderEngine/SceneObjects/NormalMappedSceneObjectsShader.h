#pragma once

#include "RenderEngine/SceneObjects/SceneObjectsShader.h"

class NormalMappedSceneObjectsShader : public SceneObjectsShader
{
public:
    NormalMappedSceneObjectsShader();

    void connectTextureUnits() override;
    void getAllUniformLocations() override;

private:
    int location_normal_texture;
};

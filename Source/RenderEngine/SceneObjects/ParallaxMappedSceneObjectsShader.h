#pragma once

#include "RenderEngine/SceneObjects/SceneObjectsShader.h"

class ParallaxMappedSceneObjectsShader : public SceneObjectsShader
{
public:
    ParallaxMappedSceneObjectsShader();

    void connectTextureUnits() override;
    void getAllUniformLocations() override;

    void loadHeightScale(float height_scale);

private:
    int location_normal_texture;
    int location_depth_texture;
    int location_height_scale;
};

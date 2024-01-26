#pragma once

#include "RenderEngine/SceneObjects/SceneObjectsShader.h"

class ImageShader : public ShaderProgram
{
public:
    ImageShader();
    virtual ~ImageShader() = default;

    void connectTextureUnits() override;
    void getAllUniformLocations() override;

private:
    int location_texture;
};

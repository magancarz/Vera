#pragma once

#include "RenderEngine/SceneObjects/SceneObjectsShader.h"

class ImageShader : public ShaderProgram
{
public:
    ImageShader();
    virtual ~ImageShader() = default;

    void connectTextureUnits() const;
    void getAllUniformLocations() override;

private:
    int location_texture;
};

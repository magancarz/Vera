#pragma once

#include "Shaders/ShaderProgram.h"

class HorizontalBlurShader : public ShaderProgram
{
public:
    HorizontalBlurShader();

    void getAllUniformLocations() override;
    void connectTextureUnits() override;

private:
    int location_blurred_texture;
};

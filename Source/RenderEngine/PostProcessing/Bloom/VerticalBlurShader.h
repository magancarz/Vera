#pragma once

#include "Shaders/ShaderProgram.h"

class VerticalBlurShader : public ShaderProgram
{
public:
    VerticalBlurShader();

    void getAllUniformLocations() override;
    void connectTextureUnits() override;

private:
    int location_blurred_texture;
};

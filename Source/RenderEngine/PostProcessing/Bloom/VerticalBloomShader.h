#pragma once

#include "Shaders/ShaderProgram.h"

class VerticalBloomShader : public ShaderProgram
{
public:
    VerticalBloomShader();

    void getAllUniformLocations() override;
    void connectTextureUnits() override;

private:
    int location_blurred_texture;
};

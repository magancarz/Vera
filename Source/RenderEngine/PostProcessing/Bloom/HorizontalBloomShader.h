#pragma once

#include "Shaders/ShaderProgram.h"

class HorizontalBloomShader : public ShaderProgram
{
public:
    HorizontalBloomShader();

    void getAllUniformLocations() override;
    void connectTextureUnits() override;

private:
    int location_blurred_texture;
};

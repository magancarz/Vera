#pragma once

#include "Shaders/ShaderProgram.h"

class CombineShader : public ShaderProgram
{
public:
    CombineShader();

    void getAllUniformLocations() override;
    void connectTextureUnits() override;

private:
    int location_blurred_texture;
    int location_hdr_color_buffer;
};

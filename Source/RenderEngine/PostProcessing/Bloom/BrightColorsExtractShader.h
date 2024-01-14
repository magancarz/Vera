#pragma once

#include "Shaders/ShaderProgram.h"

class BrightColorsExtractShader : public ShaderProgram
{
public:
    BrightColorsExtractShader();

    void getAllUniformLocations() override;
    void connectTextureUnits() override;

private:
    int location_hdr_color_buffer;
};

#pragma once

#include "Shaders/ShaderProgram.h"

class HDRShader : public ShaderProgram
{
public:
    HDRShader();

    void getAllUniformLocations() override;
    void connectTextureUnits() override;

private:
    int location_hdr_buffer;
};

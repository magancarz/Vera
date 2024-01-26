#pragma once

#include "Shaders/ShaderProgram.h"

class ToneMappingShader : public ShaderProgram
{
public:
    ToneMappingShader();

    void getAllUniformLocations() override;
    void connectTextureUnits() override;

private:
    int location_hdr_buffer;
};

#pragma once

#include "Shaders/ShaderProgram.h"

class SSAOBlurShader : public ShaderProgram
{
public:
    SSAOBlurShader();

    void getAllUniformLocations() override;
    void connectTextureUnits() override;

private:
    int location_ssao_input;
};
#pragma once

#include "Shaders/StaticShader.h"

class RayTracedImageShader : public StaticShader
{
public:
    RayTracedImageShader();
    virtual ~RayTracedImageShader() = default;
};

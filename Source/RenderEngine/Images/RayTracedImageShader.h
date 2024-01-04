#pragma once

#include "RenderEngine/SceneObjects/SceneObjectsShader.h"

class RayTracedImageShader : public SceneObjectsShader
{
public:
    RayTracedImageShader();
    virtual ~RayTracedImageShader() = default;
};

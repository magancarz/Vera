#pragma once

#include "RenderEngine/SceneObjects/SceneObjectsShader.h"

class RayTracedImageShader : public ShaderProgram
{
public:
    RayTracedImageShader();
    virtual ~RayTracedImageShader() = default;

    void connectTextureUnits() const;
    void getAllUniformLocations() override;

private:
    int location_texture;
};

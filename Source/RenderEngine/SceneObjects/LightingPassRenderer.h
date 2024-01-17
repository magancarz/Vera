#pragma once

#include "Models/RawModel.h"
#include "LightingPassShader.h"
#include "Models/AssetManager.h"

class Camera;

class LightingPassRenderer
{
public:
    void render(const std::shared_ptr<Camera>& camera, const utils::Texture& ssao);

    template <typename T>
    void bindUniformBuffer(const UniformBuffer<T>& uniform_buffer)
    {
        lighting_pass_shader.bindUniformBuffer(uniform_buffer);
    }

private:
    LightingPassShader lighting_pass_shader;
};
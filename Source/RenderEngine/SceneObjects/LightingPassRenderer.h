#pragma once

#include "Models/RawModel.h"
#include "LightingPassShader.h"
#include "Models/AssetManager.h"

class LightingPassRenderer
{
public:
    LightingPassRenderer();

    void render(const utils::Texture& g_position, const utils::Texture& g_normal, const utils::Texture& g_color_spec);

    template <typename T>
    void bindUniformBuffer(const UniformBuffer<T>& uniform_buffer)
    {
        lighting_pass_shader.bindUniformBuffer(uniform_buffer);
    }

private:
    LightingPassShader lighting_pass_shader;
    RawModelAttributes quad;
    inline static const std::vector<float> quad_positions =
    {
            -1.0f, 1.0f,
            -1.0f, -1.0f,
            1.0f, 1.0f,
            1.0f, -1.0f
    };
    inline static const std::vector<float> quad_textures =
    {
            0.0f, 1.0f,
            0.0, 0.0,
            1.0, 1.0,
            1.0, 0.0,
    };
};
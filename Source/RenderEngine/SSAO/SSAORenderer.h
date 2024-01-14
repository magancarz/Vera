#pragma once

#include <vector>
#include <random>

#include "glm/glm.hpp"

#include "Models/RawModel.h"
#include "UniformBuffer.h"
#include "SSAOShader.h"
#include "SSAOBlurShader.h"

class SSAORenderer
{
public:
    SSAORenderer();

    void render();

    template <typename T>
    void bindUniformBuffer(const UniformBuffer<T>& uniform_buffer)
    {
        ssao_shader.bindUniformBuffer(uniform_buffer);
    }

    utils::Texture ssao_blur_color_buffer;

private:
    void generateSampleKernel();
    float lerp(float a, float b, float f);
    void generateNoise();
    void createSSAOFramebuffer();
    void createSSAOBlurFramebuffer();
    void prepareSSAOShader();

    SSAOShader ssao_shader;
    utils::FBO ssao_fbo;
    utils::Texture ssao_color_buffer;

    SSAOBlurShader ssao_blur_shader;
    utils::FBO ssao_blur_fbo;

    std::vector<glm::vec3> ssao_kernel;
    std::vector<glm::vec3> ssao_noise;
    utils::Texture noise_texture;

    std::uniform_real_distribution<float> random_floats{0.0f, 1.0f};
    std::default_random_engine generator;
};

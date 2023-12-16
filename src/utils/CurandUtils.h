#pragma once

#include <curand_kernel.h>
#include <random>
#include <corecrt_math_defines.h>
#include <glm/glm.hpp>

__global__ void initCurandState(curandState* curand_state, unsigned long long seed);

__device__ inline float randomFloat(curandState* curand_state)
{
    float random_float = curand_uniform(curand_state);
    if (random_float >= 1.0f)
    {
        random_float -= 0.00001f;
    }
    return random_float;
}

__device__ inline float randomFloat(curandState* curand_state, float a, float b)
{
    const float random_float = randomFloat(curand_state);
    const float temp = random_float * (b - a) + a;
    assert(temp >= a);
    assert(temp < b);
    return temp;
}

__device__ inline int randomInt(curandState* curand_state, int a, int b)
{
    const float out = randomFloat(curand_state, static_cast<float>(a), static_cast<float>(b));
    const int temp = static_cast<int>(glm::floor(out));
    return temp;
}

__device__ inline glm::vec3 randomCosineDirection(curandState* curand_state)
{
    const auto r1 = randomFloat(curand_state);
    const auto r2 = randomFloat(curand_state);
    auto z = sqrt(1.f - r2);

    const auto phi = 2.f * 3.14159265358979323846f * r1;
    auto x = cos(phi) * sqrt(r2);
    auto y = sin(phi) * sqrt(r2);

    return {x, y, z};
}

__device__ inline glm::vec3 randomInUnitHemisphere(curandState* curand_state, glm::vec3 normal)
{
    glm::vec3 p = randomCosineDirection(curand_state);
    p = normalize(p);
    p = p * glm::sign(dot(normal, p));

    return p;
}

__device__ inline glm::vec3 randomInUnitDisk(curandState* curand_state)
{
    const float r = sqrt(randomFloat(curand_state));
    const float alpha = randomFloat(curand_state) * 2.f * static_cast<float>(M_PI);
    const float x = r * cos(alpha);
    const float y = r * sin(alpha);

    return {x, y, 0};
}
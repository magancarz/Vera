#pragma once

#include <cuda_runtime.h>
#include <glm/glm.hpp>

namespace AdditionalAlgorithms
{
    glm::mat4 createTransformationMatrix(const glm::vec2& translation, const glm::vec2& scale);
    glm::mat4 createTransformationMatrix(const glm::vec3& translation, float rx, float ry, float rz, float scale);
    __host__ __device__ bool equal(float a, float b, float round_error = 0.000000000000000001f);
}

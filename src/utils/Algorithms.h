#pragma once

#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <string>

namespace Algorithms
{
    glm::mat4 createTransformationMatrix(const glm::vec3& translation, float rx, float ry, float rz, float scale);
    __host__ __device__ bool equal(float a, float b, float round_error = 0.000000000000000001f);
    std::string floatToString(float val);
    std::string vec3ToString(const glm::vec3& vec);
    std::string vec4ToString(const glm::vec4& vec);
}

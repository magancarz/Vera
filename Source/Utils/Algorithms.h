#pragma once

#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <string>
#include <vector>
#include <memory>

namespace Algorithms
{
    glm::mat4 createTransformationMatrix(const glm::vec3& translation, const glm::vec3& rotation, const float scale);
    __host__ __device__ bool equal(float a, float b, float round_error = 0.000000000000000001f);
    std::string floatToString(float val);
    std::string vec3ToString(const glm::vec3& vec);
    std::string vec4ToString(const glm::vec4& vec);

    template <typename T>
    void removeExpiredWeakPointers(std::vector<std::weak_ptr<T>>& vector)
    {
        erase_if(vector, [&](const std::weak_ptr<T>& ptr)
        {
           return ptr.expired();
        });
    }
}

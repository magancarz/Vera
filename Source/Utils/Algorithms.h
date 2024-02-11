#pragma once

#include <glm/glm.hpp>
#include <string>
#include <vector>
#include <memory>

namespace Algorithms
{
    glm::mat4 createTransformationMatrix(const glm::vec3& translation, const glm::vec3& rotation, const float scale);
    bool equal(float a, float b, float round_error = 0.000000000000000001f);
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

    // from: https://stackoverflow.com/a/57595105
    template <typename T, typename... Rest>
    void hashCombine(std::size_t& seed, const T& v, const Rest&... rest)
    {
        seed ^= std::hash<T>{}(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        (hashCombine(seed, rest), ...);
    };
}

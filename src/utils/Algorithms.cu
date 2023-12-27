#include "Algorithms.h"

#include <glm/gtc/type_ptr.hpp>

namespace Algorithms
{
    glm::mat4 createTransformationMatrix(const glm::vec3& translation, const float rx, const float ry, const float rz,
                                         const float scale)
    {
        glm::mat4 matrix = translate(glm::mat4(1.0f), translation);
        matrix = rotate(matrix, glm::radians(rx), glm::vec3(1, 0, 0));
        matrix = rotate(matrix, glm::radians(ry), glm::vec3(0, 1, 0));
        matrix = rotate(matrix, glm::radians(rz), glm::vec3(0, 0, 1));
        matrix = glm::scale(matrix, glm::vec3(scale, scale, scale));

        return matrix;
    }

    __host__ __device__ bool equal(float a, float b, float round_error)
    {
        return a >= b - round_error && a <= b + round_error;   
    }

    std::string Algorithms::floatToString(float val)
    {
        return std::to_string(val);
    }

    std::string Algorithms::vec3ToString(const glm::vec3& vec)
    {
        return std::to_string(vec.x) + " " + std::to_string(vec.y) + " " + std::to_string(vec.z);
    }

    std::string Algorithms::vec4ToString(const glm::vec4& vec)
    {
        return std::to_string(vec.x) + " " + std::to_string(vec.y) + " " + std::to_string(vec.z) + " " + std::to_string(vec.w);
    }
}
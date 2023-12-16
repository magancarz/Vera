#include "AdditionalAlgorithms.h"

#include <glm/gtc/type_ptr.hpp>

namespace AdditionalAlgorithms
{
    glm::mat4 createTransformationMatrix(const glm::vec2& translation, const glm::vec2& scale)
    {
        glm::mat4 matrix(1.0f);
        matrix = translate(matrix, glm::vec3(translation, 0.0f));
        matrix = glm::scale(matrix, glm::vec3(scale, 1.0f));

        return matrix;
    }

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
}
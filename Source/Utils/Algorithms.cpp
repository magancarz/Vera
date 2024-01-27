#include "Algorithms.h"

#include <glm/gtc/type_ptr.hpp>

namespace Algorithms
{
    glm::mat4 createTransformationMatrix(const glm::vec3& translation, const glm::vec3& rotation, const float scale)
    {
        glm::mat4 matrix = translate(glm::mat4(1.0f), translation);
        matrix = rotate(matrix, glm::radians(rotation.x), glm::vec3(1, 0, 0));
        matrix = rotate(matrix, glm::radians(rotation.y), glm::vec3(0, 1, 0));
        matrix = rotate(matrix, glm::radians(rotation.z), glm::vec3(0, 0, 1));
        matrix = glm::scale(matrix, glm::vec3(scale, scale, scale));

        return matrix;
    }

    bool equal(float a, float b, float round_error)
    {
        return a >= b - round_error && a <= b + round_error;   
    }
}
#pragma once

#include <glm/glm.hpp>

#include <string>
#include <sstream>

struct TriangleMeshInfo
{
    std::string object_name;
    std::string model_name;
    std::string material_name;
    glm::vec3 position;
    glm::vec3 rotation;
    float scale;

    static TriangleMeshInfo fromString(const std::string& str)
    {
        std::stringstream iss(str);
        std::string prefix;
        iss >> prefix;

        std::string object_name, model_name, material_name, position_x, position_y, position_z,
                rotation_x, rotation_y, rotation_z, scale;
        iss >> object_name;
        iss >> model_name;
        iss >> material_name;
        iss >> position_x;
        iss >> position_y;
        iss >> position_z;
        glm::vec3 position{std::stof(position_x), std::stof(position_y), std::stof(position_z)};
        iss >> rotation_x;
        iss >> rotation_y;
        iss >> rotation_z;
        glm::vec3 rotation{std::stof(rotation_x), std::stof(rotation_y), std::stof(rotation_z)};
        iss >> scale;
        const float fscale = std::stof(scale);
        return {object_name, model_name, material_name, position, rotation, fscale};
    }

    [[nodiscard]] std::string toString() const
    {
        return object_name + " "
                + model_name + " "
                + material_name + " "
                + std::to_string(position.x) + " "
                + std::to_string(position.y) + " "
                + std::to_string(position.z) + " "
                + std::to_string(rotation.x) + " "
                + std::to_string(rotation.y) + " "
                + std::to_string(rotation.z) + " "
                + std::to_string(scale);
    }
};
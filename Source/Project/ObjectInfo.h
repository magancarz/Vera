#pragma once

#include <glm/glm.hpp>

#include <string>
#include <sstream>

#include "glm/ext/quaternion_trigonometric.hpp"

struct ObjectInfo
{
    std::string object_name;
    std::string mesh_name;
    std::string material_name;
    glm::vec3 position;
    glm::vec3 rotation;
    float scale;

    static ObjectInfo fromString(const std::string& str)
    {
        std::stringstream iss(str);

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
        glm::vec3 rotation{glm::radians(std::stof(rotation_x)), glm::radians(std::stof(rotation_y)), glm::radians(std::stof(rotation_z))};
        iss >> scale;
        const float fscale = std::stof(scale);
        return {object_name, model_name, material_name, position, rotation, fscale};
    }

    [[nodiscard]] std::string toString() const
    {
        return object_name + " "
            + mesh_name + " "
            + material_name + " "
            + std::to_string(position.x) + " "
            + std::to_string(position.y) + " "
            + std::to_string(position.z) + " "
            + std::to_string(glm::degrees(rotation.x)) + " "
            + std::to_string(glm::degrees(rotation.y)) + " "
            + std::to_string(glm::degrees(rotation.z)) + " "
            + std::to_string(scale);
    }
};

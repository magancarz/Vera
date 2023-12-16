#pragma once

#include <glm/glm.hpp>

#include <string>

struct ObjectInfo
{
	std::string object_name;
	std::string model_name;
	std::string material_name;
	glm::vec3 position;
	glm::vec3 rotation;
	float scale;
};
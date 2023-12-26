#pragma once

#include <string>

struct MaterialParameters
{
	std::string color_texture_name;
	std::string normal_map_texture_name;
	std::string specular_map_texture_name;
	float brightness{ 0.f };
	float fuzziness{ 1.f };
	float refractive_index{ 1.f };
};

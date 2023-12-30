#pragma once

class Shape;

struct HitRecord
{
	Shape** intersected_shape;
	bool did_hit_anything = false;
	glm::vec3 hit_point;
	float t = 0.0f;
	glm::vec3 normal;
	glm::vec2 uv;
	bool front_face;
};

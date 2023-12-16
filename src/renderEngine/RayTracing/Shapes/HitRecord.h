#pragma once

struct HitRecord
{
    class Object* parent_object;
    unsigned int triangle_id;
	class Material* material;
	glm::vec3 color;
	glm::vec3 hit_point;
	glm::vec3 normal;
	glm::vec2 uv;
	float t = 0.0f;
	bool did_hit_anything = false;
	bool front_face;

	__device__ void setFaceNormal(const Ray& ray, const glm::vec3& outward_normal)
	{
		front_face = glm::dot(ray.direction, outward_normal) < 0.f;
		normal = front_face ? outward_normal : -outward_normal;
	}
};

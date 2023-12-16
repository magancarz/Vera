#pragma once

#include "Geometry/Bounds.h"
#include "RenderEngine/RayTracing/Ray.h"
#include "Utils/AdditionalAlgorithms.h"
#include "HitRecord.h"

class Triangle
{
public:
	void prepare(const struct TriangleData& triangle_data);

    __device__ HitRecord intersects(const Ray* r) const;

	__device__ float pdfValue(class BVHTreeTraverser* intersection_accelerator_tree_traverser, const glm::vec3& origin, const glm::vec3& direction);
	__device__ glm::vec3 random(curandState* curand_state, const glm::vec3& origin);

	void setTransform(glm::mat4* object_to_world, glm::mat4* world_to_object);
	void resetTransform();
	void applyTransform(const glm::mat4& transform);

	void calculateObjectBounds();
	void calculateWorldBounds();
	void calculateShapeSurfaceArea();

	__host__ __device__ glm::vec3 getNormalAt(const glm::vec3& barycentric_coordinates) const;
	__host__ __device__ glm::vec3 getNormalAt(float u, float v, float w) const;

	__host__ __device__ bool isEmittingLight() const;

	Object* parent;
	unsigned int id;

	Material* material;

	glm::mat4* object_to_world;
	glm::mat4* world_to_object;

	Bounds3f object_bounds;
	Bounds3f world_bounds;
	float area;

    glm::vec3 x;
    glm::vec3 y;
    glm::vec3 z;
	glm::vec2 uv_x;
	glm::vec2 uv_y;
	glm::vec2 uv_z;
    glm::vec3 normal_x;
    glm::vec3 normal_y;
    glm::vec3 normal_z;
	glm::vec3 average_normal;

private:
	void computeAverageNormal();
	bool areTriangleNormalsValid() const;
	void transformNormal(const glm::mat4& transform);
};

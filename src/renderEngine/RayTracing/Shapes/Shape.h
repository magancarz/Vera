#pragma once

#include "Geometry/Bounds.h"
#include "RenderEngine/RayTracing/Ray.h"
#include "HitRecord.h"

class Object;
class Material;

class Shape {
public:
    __device__ Shape(Object* parent, size_t id, Material* material);

	__device__ virtual HitRecord intersects(const Ray* r) const = 0;

	__device__ virtual float calculatePDFValue(const glm::vec3& origin, const glm::vec3& direction) = 0;
	__device__ virtual glm::vec3 random(curandState* curand_state, const glm::vec3& origin) = 0;

	__device__ void setTransform(glm::mat4* object_to_world, glm::mat4* world_to_object);
	__device__ void resetTransform();
	__device__ virtual void applyTransform(const glm::mat4& transform) = 0;

	__device__ virtual void calculateObjectBounds() = 0;
	__device__ virtual void calculateWorldBounds() = 0;
	__device__ virtual void calculateShapeSurfaceArea() = 0;

	__host__ __device__ virtual bool isEmittingLight() const = 0;

    Object* parent{nullptr};
	size_t id{0};

	Material* material{nullptr};

	glm::mat4* object_to_world{nullptr};
	glm::mat4* world_to_object{nullptr};

	Bounds3f object_bounds;
	Bounds3f world_bounds;
	float area{0.f};
};

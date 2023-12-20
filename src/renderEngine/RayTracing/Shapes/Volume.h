#pragma once

#include "Shape.h"

class Volume : public Shape {
public:
    __device__ Volume(Object* parent, size_t id, Material* material, Bounds3f bounds, float density);
	__device__ virtual ~Volume() = default;

	__device__ HitRecord checkRayIntersection(const Ray* r) const override;
	__device__ bool scatter(const Ray* r_in, const HitRecord* rec, ScatterRecord* scatter_record) override;

	__device__ void applyTransform(const glm::mat4& transform) override;

	__device__ void calculateObjectBounds() override;
	__device__ void calculateWorldBounds() override;
	__device__ void calculateShapeSurfaceArea() override;

private:
    float negative_inv_density{-1.f};
};
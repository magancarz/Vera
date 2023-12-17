#pragma once

#include "Shape.h"
#include "Utils/AdditionalAlgorithms.h"
#include "Models/Vertex.h"

class Triangle : public Shape
{
public:
	__device__ Triangle(Object* parent, size_t id, Material* material, const struct TriangleData& triangle_data);

    __device__ HitRecord checkRayIntersection(const Ray* r) const override;

	__device__ float calculatePDFValueOfEmittedLight(const glm::vec3& origin, const glm::vec3& direction) override;
	__device__ glm::vec3 randomDirectionAtShape(curandState* curand_state, const glm::vec3& origin) override;

	__device__ void applyTransform(const glm::mat4& transform) override;

	__device__ bool isEmittingLight() const override;

private:
    
	__device__ void calculateObjectBounds() override;
	__device__ void calculateWorldBounds() override;
	__device__ void calculateShapeSurfaceArea() override;

    __device__ void computeAverageNormal();
	__device__ void transformNormal(const glm::mat4& transform);
	__device__ bool areTriangleNormalsValid() const;

	__device__ glm::vec3 getNormalAt(const glm::vec3& barycentric_coordinates) const;
	__device__ glm::vec3 getNormalAt(float u, float v, float w) const;

    Vertex x, y, z;
	glm::vec3 average_normal;
};

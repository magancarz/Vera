#pragma once

#include "Shape.h"
#include "Utils/AdditionalAlgorithms.h"
#include "Models/Vertex.h"

class Triangle : public Shape
{
public:
	__device__ Triangle(Object* in_parent, size_t in_id, Material* in_material, const struct TriangleData& triangle_data);
	__device__ virtual ~Triangle() = default;

    __device__ HitRecord checkRayIntersection(const Ray* r) const override;
	__device__ bool scatter(const Ray* r_in, const HitRecord* rec, ScatterRecord* scatter_record) override;
	__device__ float scatteringPDF(const HitRecord* rec, const Ray* scattered) const override;
	__device__ glm::vec3 emitted(const glm::vec2& uv) override;

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

	__device__ glm::vec3 getNormalAt(float u, float v, float w) const;

    Vertex x, y, z;
	glm::vec3 average_normal{};
};

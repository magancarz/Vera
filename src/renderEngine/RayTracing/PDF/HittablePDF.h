#pragma once

#include "PDF.h"

class Shape;
class Ray;
struct HitRecord;
class BVHTreeTraverser;
class Triangle;
class IntersectionAcceleratorTreeTraverser;

class HittablePDF : public PDF
{
public:
    __device__ HittablePDF(curandState* curand_state, BVHTreeTraverser* intersection_accelerator_tree_traverser, Shape** triangles, int num_of_triangles);

    __device__ float value(const glm::vec3& direction) const override;
    __device__ glm::vec3 generate() const override;
    __device__ void changeHitRecord(HitRecord* val) { hit_record = val; }

private:
    void shuffleLightSources(Shape** shuffled_light_sources) const;
    glm::vec3 directRayToRandomLightSourceFromScene() const;

    BVHTreeTraverser* intersection_accelerator_tree_traverser;
    Shape** triangles;
    int num_of_triangles;
    HitRecord* hit_record;
};

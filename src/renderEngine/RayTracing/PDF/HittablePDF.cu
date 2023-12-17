#include "HittablePDF.h"

#include "RenderEngine/RayTracing/IntersectionAccelerators/BVHTreeTraverser.h"
#include "Utils/CurandUtils.h"
#include "RenderEngine/RayTracing/Shapes/Triangle.h"

__device__ HittablePDF::HittablePDF(curandState* curand_state, BVHTreeTraverser* intersection_accelerator_tree_traverser, Shape** triangles, int num_of_triangles)
    : PDF(curand_state), intersection_accelerator_tree_traverser(intersection_accelerator_tree_traverser), triangles(triangles), num_of_triangles(num_of_triangles) {}

__device__ float HittablePDF::value(const glm::vec3& direction) const
{
    float out_pdf = 0.f;
    const float weight = 1.f / static_cast<float>(num_of_triangles);
    for (int i = 0; i < num_of_triangles; ++i)
    {
        out_pdf += triangles[i]->calculatePDFValueOfEmittedLight(hit_record->hit_point, direction) * weight;
    }
    return out_pdf;
}

__device__ glm::vec3 HittablePDF::generate() const
{
    return directRayToRandomLightSourceFromScene();
}

__device__ void HittablePDF::shuffleLightSources(Shape** shuffled_light_sources) const
{
    for (int i = num_of_triangles - 1; i > 0; i--) {
        const int j = randomInt(curand_state, 0, num_of_triangles);
        Shape* temp = shuffled_light_sources[i];
        shuffled_light_sources[i] = shuffled_light_sources[j];
        shuffled_light_sources[j] = temp;
    }
}

__device__ glm::vec3 HittablePDF::directRayToRandomLightSourceFromScene() const
{
    auto shuffled_light_sources = static_cast<Shape**>(malloc(num_of_triangles * sizeof(void*)));
    memcpy(shuffled_light_sources, triangles, num_of_triangles * sizeof(void*));
    shuffleLightSources(shuffled_light_sources);

    glm::vec3 random_to_light_source;
    for (int i = 0; i < num_of_triangles; ++i)
    {
        Shape* light_source = shuffled_light_sources[i];
        random_to_light_source = light_source->randomDirectionAtShape(curand_state, hit_record->hit_point);
        Ray ray_to_random_light_source{hit_record->hit_point, random_to_light_source};
        HitRecord result = intersection_accelerator_tree_traverser->checkIntersection(&ray_to_random_light_source);
        if (result.did_hit_anything && result.triangle_id == light_source->id)
        {
            break;
        }
    }

    free(shuffled_light_sources);

    return random_to_light_source;
}
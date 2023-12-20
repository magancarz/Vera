#include "Volume.h"

#include "renderEngine/RayTracing/ScatterRecord.h"
#include "Utils/CurandUtils.h"

__device__ Volume::Volume(Object* parent, size_t id, Material* material, Bounds3f bounds, float density)
    : Shape(parent, id, material), negative_inv_density(-1.f / density)
{
    object_bounds = bounds;
}

__device__ HitRecord Volume::checkRayIntersection(const Ray* r) const
{
    float t_min, t_max;
    if (!object_bounds.intersect(r, r->inv_dir, r->is_dir_neg, &t_min, &t_max))
    {
        return {};
    }

    t_min = t_min < 0 ? 0 : t_min;

    const float distance_inside_boundary = t_max - t_min;
    const float hit_distance = negative_inv_density * log(randomFloat(r->curand_state));

    if (hit_distance > distance_inside_boundary)
    {
        return {};
    }

    HitRecord hit_record{};
    hit_record.t = t_min + hit_distance;
    hit_record.hit_point = r->pointAtParameter(hit_record.t);
    hit_record.normal = glm::vec3{1, 0, 0};
    hit_record.did_hit_anything = true;

    return hit_record;
}

bool Volume::scatter(const Ray* r_in, const HitRecord* rec, ScatterRecord* scatter_record)
{
    scatter_record->color = glm::vec3{0.7f, 0.7f, 0.7f};
    scatter_record->specular_ray = Ray{rec->hit_point, randomCosineDirection(r_in->curand_state)};
}

__device__ void Volume::applyTransform(const glm::mat4& transform)
{
    object_bounds.min = glm::vec3(transform * glm::vec4(object_bounds.min, 1.0f));
    object_bounds.max = glm::vec3(transform * glm::vec4(object_bounds.max, 1.0f));
}

__device__ void Volume::calculateObjectBounds() {}

__device__ void Volume::calculateWorldBounds() {}

__device__ void Volume::calculateShapeSurfaceArea()
{
    area = object_bounds.surfaceArea();
}
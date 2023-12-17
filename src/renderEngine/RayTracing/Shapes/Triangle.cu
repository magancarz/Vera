#include "Triangle.h"

#include "Models/TriangleData.h"
#include "Materials/Material.h"
#include "Utils/CurandUtils.h"

__device__ Triangle::Triangle(Object* parent, size_t id, Material* material, const TriangleData& triangle_data)
    : Shape(parent, id, material)
{
    object_to_world = nullptr;
    world_to_object = nullptr;

    x = triangle_data.vertices[0];
    y = triangle_data.vertices[1];
    z = triangle_data.vertices[2];

    computeAverageNormal();
}

__device__ HitRecord Triangle::checkRayIntersection(const Ray* r) const
{
    constexpr float EPSILON = 0.00000001f;
    constexpr float MIN_DST = 0.0001f;
    glm::vec3 edge_xy = y.position - x.position;
    glm::vec3 edge_xz = z.position - x.position;
    glm::vec3 ao = r->origin - x.position;
    glm::vec3 normal = cross(edge_xy, edge_xz);

    float determinant = dot(-r->direction, normal);
    if (fabs(determinant) < EPSILON) return {};
    float inv_det = 1.f / determinant;

    float dst = dot(ao, normal) * inv_det;
    float u = dot(-r->direction, cross(ao, edge_xz)) * inv_det;
    float v = dot(-r->direction, cross(edge_xy, ao)) * inv_det;
    float w = 1.f - u - v;

    HitRecord hit_record_result{};
    if (dst >= MIN_DST && u >= 0 && v >= 0 && w >= 0)
    {
        hit_record_result.uv = w * x.texture_coordinate + u * y.texture_coordinate + v * z.texture_coordinate;
        if (material->getColorAlphaValue(hit_record_result.uv) < 0.5f)
        {
            return hit_record_result;
        }
        hit_record_result.hit_point = r->origin + r->direction * dst;
        hit_record_result.front_face = determinant > 0;
        hit_record_result.t = dst;
        hit_record_result.did_hit_anything = true;
        hit_record_result.parent_object = parent;
        hit_record_result.triangle_id = id;
        hit_record_result.material = material;
        hit_record_result.color = material->getColor(hit_record_result.uv);
        glm::vec3 new_normal;
    	if (material->hasNormalMap())
        {
            new_normal = w * material->getNormal(x.texture_coordinate) + u * material->getNormal(y.texture_coordinate) + v * material->getNormal(z.texture_coordinate);
        }
    	else
        {
            new_normal = getNormalAt(w, u, v);
        }
        hit_record_result.normal = determinant > 0 ? normalize(new_normal) : normalize(-new_normal);
    }

    return hit_record_result;
}

__device__ float Triangle::calculatePDFValueOfEmittedLight(const glm::vec3& origin, const glm::vec3& direction)
{
    Ray ray{origin, direction};
    const auto rec = checkRayIntersection(&ray);
    if (rec.triangle_id != id)
    {
        return 0;
    }

    float cosine = fabs(dot(direction, rec.normal));
    const float distance_squared = rec.t * rec.t;
    cosine = cosine < 0.00000001f ? 0.00000001f : cosine;

    return distance_squared / (cosine * area);
}

__device__ glm::vec3 Triangle::randomDirectionAtShape(curandState* curand_state, const glm::vec3& origin)
{
    float u = randomFloat(curand_state);
    float v = randomFloat(curand_state);

    if (u + v > 1.f)
    {
        u = 1.f - u;
        v = 1.f - v;
    }

    const float w = 1.f - u - v;

    const glm::vec3 random_point_on_a_triangle = u * x.position + v * y.position + w * z.position;

    return glm::normalize(random_point_on_a_triangle - origin);
}

__device__ void Triangle::applyTransform(const glm::mat4& transform)
{
    x.position = glm::vec3(transform * glm::vec4(x.position, 1.0f));
    y.position = glm::vec3(transform * glm::vec4(y.position, 1.0f));
    z.position = glm::vec3(transform * glm::vec4(z.position, 1.0f));
    transformNormal(transform);
}

__device__ void Triangle::calculateObjectBounds()
{
    object_bounds = boundsFromUnion(x.position, y.position, z.position);
}

__device__ void Triangle::calculateWorldBounds()
{
    const glm::vec3 world_x = glm::vec3(*object_to_world * glm::vec4(x.position, 1.0f));
    const glm::vec3 world_y = glm::vec3(*object_to_world * glm::vec4(y.position, 1.0f));
    const glm::vec3 world_z = glm::vec3(*object_to_world * glm::vec4(z.position, 1.0f));
    world_bounds = boundsFromUnion(world_x, world_y, world_z);
}

__device__ void Triangle::calculateShapeSurfaceArea()
{
    area = glm::length(glm::cross(glm::vec3{y.position - x.position}, glm::vec3{z.position - x.position})) / 2.f;
}

__host__ __device__ bool Triangle::isEmittingLight() const
{
    return material->getSpecularValue(x.texture_coordinate).g > 0.5f ||
        material->getSpecularValue(y.texture_coordinate).g > 0.5f ||
        material->getSpecularValue(z.texture_coordinate).g > 0.5f;
}

__host__ __device__ glm::vec3 Triangle::getNormalAt(const glm::vec3& barycentric_coordinates) const
{
    return x.normal * barycentric_coordinates.x + y.normal * barycentric_coordinates.y + z.normal * barycentric_coordinates.z;
}

__host__ __device__ glm::vec3 Triangle::getNormalAt(float u, float v, float w) const
{
    return x.normal * u + y.normal * v + z.normal * w;
}

__device__ void Triangle::computeAverageNormal()
{
    if (!areTriangleNormalsValid())
    {
        average_normal = normalize(glm::cross(y.position - x.position, z.position - x.position));
        x.normal = y.normal = z.normal = average_normal;
        return;
    }

    constexpr float one_third = 1.f / 3;
    average_normal = getNormalAt(one_third, one_third, one_third);
}

__device__ bool Triangle::areTriangleNormalsValid() const
{
    return glm::length(x.normal) < 0.0001f && glm::length(y.normal) < 0.0001f && glm::length(z.normal) < 0.0001f;
}

__device__ void Triangle::transformNormal(const glm::mat4& transform)
{
    auto transposed_inverse = glm::mat3{transform};
    transposed_inverse = glm::transpose(glm::inverse(transposed_inverse));

    x.normal = glm::normalize(transposed_inverse * x.normal);
    y.normal = glm::normalize(transposed_inverse * y.normal);
    z.normal = glm::normalize(transposed_inverse * z.normal);
    average_normal = glm::normalize(transposed_inverse * average_normal);
}

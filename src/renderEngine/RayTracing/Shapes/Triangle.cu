#include "Triangle.h"

#include "Models/TriangleData.h"
#include "Materials/Material.h"
#include "Utils/CurandUtils.h"

void Triangle::prepare(const TriangleData& triangle_data)
{
    object_to_world = nullptr;
    world_to_object = nullptr;

    x = triangle_data.vertices[0].position;
    y = triangle_data.vertices[1].position;
    z = triangle_data.vertices[2].position;

    uv_x = triangle_data.vertices[0].texture_coordinate;
    uv_y = triangle_data.vertices[1].texture_coordinate;
    uv_z = triangle_data.vertices[2].texture_coordinate;

    normal_x = triangle_data.vertices[0].normal;
    normal_y = triangle_data.vertices[1].normal;
    normal_z = triangle_data.vertices[2].normal;
    computeAverageNormal();
}

__device__ HitRecord Triangle::intersects(const Ray* r) const
{
    constexpr float EPSILON = 0.00000001f;
    constexpr float MIN_DST = 0.0001f;
    glm::vec3 edge_xy = y - x;
    glm::vec3 edge_xz = z - x;
    glm::vec3 ao = r->origin - x;
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
        hit_record_result.uv = w * uv_x + u * uv_y + v * uv_z;
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
            new_normal = w * material->getNormal(uv_x) + u * material->getNormal(uv_y) + v * material->getNormal(uv_z);
        }
    	else
        {
            new_normal = getNormalAt(w, u, v);
        }
        hit_record_result.normal = determinant > 0 ? normalize(new_normal) : normalize(-new_normal);
    }

    return hit_record_result;
}

__device__ float Triangle::calculatePDFValue(const glm::vec3& origin, const glm::vec3& direction)
{
    Ray ray{origin, direction};
    const auto rec = intersects(&ray);
    if (rec.triangle_id != id)
    {
        return 0;
    }

    float cosine = fabs(dot(direction, rec.normal));
    const float distance_squared = rec.t * rec.t;
    cosine = cosine < 0.00000001f ? 0.00000001f : cosine;

    return distance_squared / (cosine * area);
}

__device__ glm::vec3 Triangle::random(curandState* curand_state, const glm::vec3& origin)
{
    float u = randomFloat(curand_state);
    float v = randomFloat(curand_state);

    if (u + v > 1.f)
    {
        u = 1.f - u;
        v = 1.f - v;
    }

    const float w = 1.f - u - v;

    const glm::vec3 random_point_on_a_triangle = u * x + v * y + w * z;

    return glm::normalize(random_point_on_a_triangle - origin);
}

void Triangle::applyTransform(const glm::mat4& transform)
{
    x = glm::vec3(transform * glm::vec4(x, 1.0f));
    y = glm::vec3(transform * glm::vec4(y, 1.0f));
    z = glm::vec3(transform * glm::vec4(z, 1.0f));
    transformNormal(transform);
}

void Triangle::calculateObjectBounds()
{
    object_bounds = boundsFromUnion(x, y, z);
}

void Triangle::calculateWorldBounds()
{
    const glm::vec3 world_x = glm::vec3(*object_to_world * glm::vec4(x, 1.0f));
    const glm::vec3 world_y = glm::vec3(*object_to_world * glm::vec4(y, 1.0f));
    const glm::vec3 world_z = glm::vec3(*object_to_world * glm::vec4(z, 1.0f));
    world_bounds = boundsFromUnion(world_x, world_y, world_z);
}

void Triangle::calculateShapeSurfaceArea()
{
    area = glm::length(glm::cross(glm::vec3{y - x}, glm::vec3{z - x})) / 2.f;
}

void Triangle::setTransform(glm::mat4* object_to_world_val, glm::mat4* world_to_object_val)
{
    this->object_to_world = object_to_world_val;
    this->world_to_object = world_to_object_val;
    calculateObjectBounds();
    calculateWorldBounds();
    calculateShapeSurfaceArea();
    applyTransform(*this->object_to_world);
}

void Triangle::resetTransform()
{
    if (world_to_object != nullptr)
    {
        applyTransform(*world_to_object);
    }
}

__host__ __device__ bool Triangle::isEmittingLight() const
{
    return material->getSpecularValue(uv_x).g > 0.5f ||
        material->getSpecularValue(uv_y).g > 0.5f ||
        material->getSpecularValue(uv_z).g > 0.5f;
}

__host__ __device__ glm::vec3 Triangle::getNormalAt(const glm::vec3& barycentric_coordinates) const
{
    return normal_x * barycentric_coordinates.x + normal_y * barycentric_coordinates.y + normal_z * barycentric_coordinates.z;
}

__host__ __device__ glm::vec3 Triangle::getNormalAt(float u, float v, float w) const
{
    return normal_x * u + normal_y * v + normal_z * w;
}

void Triangle::computeAverageNormal()
{
    if (!areTriangleNormalsValid())
    {
        average_normal = normalize(glm::cross(y - x, z - x));
        normal_x = normal_y = normal_z = average_normal;
        return;
    }

    constexpr float one_third = 1.f / 3;
    average_normal = getNormalAt(one_third, one_third, one_third);
}

bool Triangle::areTriangleNormalsValid() const
{
    return glm::length(normal_x) < 0.0001f && glm::length(normal_y) < 0.0001f && glm::length(normal_z) < 0.0001f;
}

void Triangle::transformNormal(const glm::mat4& transform)
{
    auto transposed_inverse = glm::mat3{transform};
    transposed_inverse = glm::transpose(glm::inverse(transposed_inverse));

    normal_x = glm::normalize(transposed_inverse * normal_x);
    normal_y = glm::normalize(transposed_inverse * normal_y);
    normal_z = glm::normalize(transposed_inverse * normal_z);
    average_normal = glm::normalize(transposed_inverse * average_normal);
}

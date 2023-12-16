#pragma once

#include "RenderEngine/RayTracing/Ray.h"

class Bounds3f
{
public:
    __host__ __device__ Bounds3f()
    {
        float min_num = FLT_MIN;
        float max_num = FLT_MAX;
        min = glm::vec3(max_num, max_num, max_num);
        max = glm::vec3(min_num, min_num, min_num);
    }

    __host__ __device__ Bounds3f(const glm::vec3& p)
        : min(p), max(p) {}

    __host__ __device__ Bounds3f(const glm::vec3& p1, const glm::vec3& p2)
        : min(glm::min(p1.x, p2.x), glm::min(p1.y, p2.y), glm::min(p1.z, p2.z)),
          max(glm::max(p1.x, p2.x), glm::max(p1.y, p2.y), glm::max(p1.z, p2.z)) {}

    __host__ __device__ const glm::vec3& operator[](size_t i) const
    {
        assert(i <= 1);
        if (i == 0) return min;
        return max;
    }

    __host__ __device__ const glm::vec3& operator[](size_t i)
    {
        assert(i <= 1);
        if (i == 0) return min;
        return max;
    }

    __host__ __device__ glm::vec3 corner(int corner) const
    {
        return glm::vec3((*this)[(corner & 1)].x,
                         (*this)[(corner & 2) ? 1 : 0].y,
                         (*this)[(corner & 4) ? 1 : 0].z);
    }

    __host__ __device__ glm::vec3 diagonal() const
    {
        return max - min;
    }

    __host__ __device__ float surfaceArea() const
    {
        glm::vec3 d = diagonal();
        return 2 * (d.x * d.y + d.x * d.z + d.y * d.z);
    }

    __host__ __device__ float volume() const
    {
        glm::vec3 d = diagonal();
        return d.x * d.y * d.z;
    }

    __host__ __device__ int maximumExtent() const
    {
        glm::vec3 d = diagonal();
        if (d.x > d.y && d.x > d.z)
            return 0;
        if (d.y > d.z)
            return 1;
        return 2;
    }

    __host__ __device__ glm::vec3 offset(const glm::vec3& p) const
    {
        glm::vec3 o = p - min;
        if (max.x > min.x) o.x /= max.x - min.x;
        if (max.y > min.y) o.y /= max.y - min.y;
        if (max.z > min.z) o.z /= max.z - min.z;
        return o;
    }

    __host__ __device__ bool intersect(const Ray* ray, const glm::vec3& inv_dir, const bool dir_is_neg[3], float* out_t_min) const
    {
        const Bounds3f &bounds = *this;
        float t_min = (bounds[dir_is_neg[0]].x - ray->origin.x) * inv_dir.x;
        float t_max = (bounds[1.f - dir_is_neg[0]].x - ray->origin.x) * inv_dir.x;
        float ty_min = (bounds[dir_is_neg[1]].y - ray->origin.y) * inv_dir.y;
        float ty_max = (bounds[1.f - dir_is_neg[1]].y - ray->origin.y) * inv_dir.y;
        
        if (t_min > ty_max || ty_min > t_max) return false;
        if (ty_min > t_min) t_min = ty_min;
        if (ty_max < t_max) t_max = ty_max;

        float tz_min = (bounds[dir_is_neg[2]].z - ray->origin.z) * inv_dir.z;
        float tz_max = (bounds[1.f - dir_is_neg[2]].z - ray->origin.z) * inv_dir.z;
        
        if (t_min > tz_max || tz_min > t_max) return false;
        if (tz_min > t_min) t_min = tz_min;
        if (tz_max < t_max) t_max = tz_max;
        *out_t_min = t_min;
        return (t_min < ray->max) && (t_max > 0);
    }

    glm::vec3 min, max;
};

__host__ __device__ inline Bounds3f boundsFromUnion(const Bounds3f& b, const glm::vec3& p)
{
    return Bounds3f(glm::vec3(glm::min(b.min.x, p.x),
                                glm::min(b.min.y, p.y),
                                glm::min(b.min.z, p.z)),
                      glm::vec3(glm::max(b.max.x, p.x),
                                glm::max(b.max.y, p.y),
                                glm::max(b.max.z, p.z)));
}

__host__ __device__ inline Bounds3f boundsFromUnion(const glm::vec3& p0, const glm::vec3& p1, const glm::vec3& p2)
{
    return Bounds3f(glm::vec3(glm::min(p0.x, glm::min(p1.x, p2.x)),
                              glm::min(p0.y, glm::min(p1.y, p2.y)),
                              glm::min(p0.z, glm::min(p1.z, p2.z))),
                    glm::vec3(glm::max(p0.x, glm::max(p1.x, p2.x)),
                              glm::max(p0.y, glm::max(p1.y, p2.y)),
                              glm::max(p0.z, glm::max(p1.z, p2.z))));
}

__host__ __device__ inline Bounds3f boundsFromUnion(const Bounds3f& b1, const Bounds3f& b2)
{
    return Bounds3f(glm::vec3(glm::min(b1.min.x, b2.min.x),
                                glm::min(b1.min.y, b2.min.y),
                                glm::min(b1.min.z, b2.min.z)),
                      glm::vec3(glm::max(b1.max.x, b2.max.x),
                                glm::max(b1.max.y, b2.max.y),
                                glm::max(b1.max.z, b2.max.z)));
}

__host__ __device__ inline Bounds3f intersect(const Bounds3f& b1, const Bounds3f& b2)
{
    return Bounds3f(glm::vec3(glm::max(b1.min.x, b2.min.x),
                                glm::max(b1.min.y, b2.min.y),
                                glm::max(b1.min.z, b2.min.z)),
                      glm::vec3(glm::min(b1.max.x, b2.max.x),
                                glm::min(b1.max.y, b2.max.y),
                                glm::min(b1.max.z, b2.max.z)));
}

__host__ __device__ inline bool overlaps(const Bounds3f& b1, const Bounds3f& b2)
{
    bool x = (b1.max.x >= b2.min.x) && (b1.min.x <= b2.max.x);
    bool y = (b1.max.y >= b2.min.y) && (b1.min.y <= b2.max.y);
    bool z = (b1.max.z >= b2.min.z) && (b1.min.z <= b2.max.z);
    return (x && y && z);
}

__host__ __device__ inline bool inside(const glm::vec3& p, const Bounds3f& b)
{
    return (p.x >= b.min.x && p.x <= b.max.x &&
        p.y >= b.min.y && p.y <= b.max.y &&
        p.z >= b.min.z && p.z <= b.max.z);
}

__host__ __device__ inline bool insideExclusive(const glm::vec3& p, const Bounds3f& b)
{
    return (p.x >= b.min.x && p.x < b.max.x &&
        p.y >= b.min.y && p.y < b.max.y &&
        p.z >= b.min.z && p.z < b.max.z);
}

__host__ __device__ inline Bounds3f expand(const Bounds3f& b, float delta)
{
    return Bounds3f(b.min - glm::vec3(delta, delta, delta),
                      b.max + glm::vec3(delta, delta, delta));
}

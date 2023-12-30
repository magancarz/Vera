#include "CosinePDF.h"

#include "RenderEngine/RayTracing/Ray.h"
#include "Utils/CurandUtils.h"

__device__ CosinePDF::CosinePDF(curandState* curand_state, const glm::vec3& w)
    : PDF(curand_state)
{
    uvw.buildFromVector(w);
}

__device__ float CosinePDF::value(const glm::vec3& direction) const
{
    const auto cosine = dot(normalize(direction), uvw.z);
    return (cosine < 0.f) ? 0.f : cosine / 3.14159265358979323846f;
}

__device__ glm::vec3 CosinePDF::generate() const
{
    return normalize(uvw.local(randomCosineDirection(curand_state)));
}

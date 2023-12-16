#include "MixturePDF.h"

#include "HittablePDF.h"

__device__ MixturePDF::MixturePDF(curandState* curand_state, HittablePDF first_pdf, CosinePDF second_pdf)
    : PDF(curand_state), first(first_pdf), second(second_pdf) {}

__device__ float MixturePDF::value(const glm::vec3& direction) const
{
    return 0.5f * first.value(direction) + 0.5f * second.value(direction);
}

__device__ glm::vec3 MixturePDF::generate() const
{
    if (curand_uniform(curand_state) < 0.5f)
    {
        return first.generate();
    }
    return second.generate();
}

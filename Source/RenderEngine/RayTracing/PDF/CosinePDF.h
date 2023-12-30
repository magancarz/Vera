#pragma once

#include "PDF.h"
#include "Geometry/OrthonormalBasis.h"

class CosinePDF : public PDF
{
public:
    __device__ CosinePDF() : PDF(nullptr) {}
    __device__ CosinePDF(curandState* curand_state, const glm::vec3& w);

    __device__ float value(const glm::vec3& direction) const override;
    __device__ glm::vec3 generate() const override;

private:
    OrthonormalBasis uvw;
};

#pragma once

#include <memory>

#include "CosinePDF.h"
#include "HittablePDF.h"
#include "PDF.h"

class MixturePDF : public PDF
{
public:
    __device__ MixturePDF(curandState* curand_state, HittablePDF first_pdf, CosinePDF second_pdf);

    __device__ float value(const glm::vec3& direction) const override;
    __device__ glm::vec3 generate() const override;

private:
    HittablePDF first;
    CosinePDF second;
};

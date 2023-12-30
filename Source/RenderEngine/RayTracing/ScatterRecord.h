#pragma once

#include "Ray.h"
#include "PDF/CosinePDF.h"

struct ScatterRecord
{
    Ray specular_ray;
    bool is_specular;
    CosinePDF pdf;
    glm::vec3 color{0};
};

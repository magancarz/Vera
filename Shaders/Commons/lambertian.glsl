#include "defines.glsl"

float scatteringPDFFromLambertian(vec3 normal, vec3 direction)
{
    float cosine = dot(normal, direction);
    return cosine < 0 ? 0 : cosine / PI;
}
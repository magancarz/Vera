#include "defines.glsl"

vec3 randomInUnitHemisphere(uint seed, vec3 normal)
{
    float r1 = rnd(seed);
    float r2 = rnd(seed);
    float temp = sqrt(r2 * (1.0 - r2));
    float x = cos(2.0 * PI * r1) * temp;
    float y = sin(2.0 * PI * r1) * temp;
    float z = 1.0 - 2.0 * r2;

    vec3 random = vec3(x, y, z);
    random *= sign(dot(normal, random));
    return random;
}
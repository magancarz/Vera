#include "defines.glsl"

void generateOrthonormalBasis(inout vec3 u, inout vec3 v, inout vec3 w)
{
    w = normalize(w);
    vec3 temp = abs(w.x) > 0.9 ? vec3(0, 1, 0) : vec3(1, 0, 0);
    v = normalize(cross(w, temp));
    u = cross(w, v);
}

float valueFromCosinePDF(vec3 direction, vec3 w)
{
    float cosine = dot(normalize(direction), w);
    return cosine < 0 ? 0 : cosine / PI;
}

vec3 randomCosineDirection(uint seed)
{
    float r1 = rnd(seed);
    float r2 = rnd(seed);
    float z = sqrt(1.0 - r2);

    float phi = 2.0 * PI * r1;
    float x = cos(phi) * sqrt(r2);
    float y = sin(phi) * sqrt(r2);

    return vec3(x, y, z);
}

vec3 generateRandomDirectionWithCosinePDF(uint seed, vec3 u, vec3 v, vec3 w)
{
    vec3 random_cosine_direction = randomCosineDirection(seed);
    random_cosine_direction = u * random_cosine_direction.x + v * random_cosine_direction.y + w * random_cosine_direction.z;
    return normalize(random_cosine_direction);
}
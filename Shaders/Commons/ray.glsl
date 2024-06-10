struct Ray
{
    vec3 origin;
    vec3 direction;
    vec3 color;
    int is_active;
    uint seed;
    uint depth;
};
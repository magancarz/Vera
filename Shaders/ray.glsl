struct Ray
{
    vec4 origin;
    vec4 direction;
    vec3 color;
    int is_active;
    uint seed;
    uint depth;
};
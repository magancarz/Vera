#version 330 core

in vec4 fragment_position;

uniform vec3 light_position;
uniform float far_plane;

void main()
{
    float light_distance = length(fragment_position.xyz - light_position);
    light_distance = light_distance / far_plane;
    gl_FragDepth = light_distance;
}
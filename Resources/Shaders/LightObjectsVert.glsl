#version 330 core

layout (location = 0) in vec3 position;

layout (std140) uniform TransformationMatrices
{
    mat4 model;
    mat4 view;
    mat4 proj;
};

void main(void)
{
    gl_Position = proj * view * model * vec4(position, 1.0);
}
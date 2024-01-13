#version 330 core

layout (location = 0) in vec3 position;

layout (std140) uniform TransformationMatrices
{
	mat4 model;
	mat4 view;
	mat4 proj;
};

const mat4 enlarge = mat4(vec4(1.02f, 0, 0, 0),
						  vec4(0, 1.02f, 0, 0),
						  vec4(0, 0, 1.02f, 0),
						  vec4(0, 0, 0, 1));

void main(void) {
	gl_Position = proj * view * model * enlarge * vec4(position, 1.0);
}
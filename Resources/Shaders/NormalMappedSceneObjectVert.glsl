#version 330 core

layout (location = 0) in vec3 position;
layout (location = 1) in vec2 texture_coords;
layout (location = 2) in vec3 normal;
layout (location = 3) in vec3 tangent;
layout (location = 4) in vec3 bitangent;

out vec4 fragment_world_position;
out vec2 pass_texture_coords;
out mat3 TBN;

layout (std140) uniform TransformationMatrices
{
	mat4 model;
	mat4 view;
	mat4 proj;
};

void main(void) {
	fragment_world_position = view * model * vec4(position, 1.0);
	gl_Position = proj * fragment_world_position;
	pass_texture_coords = texture_coords;

	mat3 normal_matrix = transpose(inverse(mat3(model)));
	vec3 T = normalize(mat3(normal_matrix) * tangent);
	vec3 B = normalize(mat3(normal_matrix) * bitangent);
	vec3 N = normalize(mat3(normal_matrix) * normal);
	TBN = mat3(view) * mat3(T, B, N);
}
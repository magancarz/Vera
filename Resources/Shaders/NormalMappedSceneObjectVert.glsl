#version 330 core

struct Light
{
	vec3 light_position;
	vec3 light_color;
	vec3 attenuation;
};
layout (std140) uniform LightInfos
{
	Light light;
};

layout (location = 0) in vec3 position;
layout (location = 1) in vec2 texture_coords;
layout (location = 2) in vec3 normal;
layout (location = 3) in vec3 tangent;
layout (location = 4) in vec3 bitangent;

out vec4 fragment_world_position;
out vec4 tangent_space_fragment_world_position;
out vec2 pass_texture_coords;
out vec3 tangent_space_view_position;
out vec3 tangent_space_light_position;

layout (std140) uniform TransformationMatrices
{
	mat4 model;
	mat4 view;
	mat4 proj;
};

void main(void) {
	fragment_world_position = model * vec4(position, 1.0);
	gl_Position = proj * view * fragment_world_position;
	pass_texture_coords = texture_coords;

	mat3 normal_matrix = transpose(inverse(mat3(model)));
	tangent_space_view_position = (inverse(view) * vec4(0, 0, 0, 1)).xyz;

	vec3 T = normalize(mat3(normal_matrix) * tangent);
	vec3 B = normalize(mat3(normal_matrix) * bitangent);
	vec3 N = normalize(mat3(normal_matrix) * normal);
	mat3 TBN = transpose(mat3(T, B, N));
	tangent_space_fragment_world_position = vec4(TBN * vec3(fragment_world_position), 1.0);
	tangent_space_view_position = TBN * tangent_space_view_position;
	tangent_space_light_position = TBN * light.light_position;
}
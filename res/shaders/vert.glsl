#version 330 core

layout (location = 0) in vec3 position;
layout (location = 1) in vec2 texture_coords;
layout (location = 2) in vec3 normal;
layout (location = 3) in vec3 tangent;
layout (location = 4) in vec3 bitangent;

out vec4 world_position;
out vec2 pass_texture_coords;
out vec3 to_camera_vector;
out vec3 surface_normal;
out mat3 TBN;

uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;

uniform vec4 plane;
uniform int normal_map_loaded;

void main(void) {
	world_position = model * vec4(position, 1.0);
	gl_Position = proj * view * world_position;
	pass_texture_coords = texture_coords;

	surface_normal = normalize((model * vec4(normal, 0.0)).xyz);
	to_camera_vector = normalize((inverse(view) * vec4(0, 0, 0, 1)).xyz - world_position.xyz);

	if (normal_map_loaded > 0)
	{
		mat3 normal_matrix = transpose(inverse(mat3(model)));
		vec3 T = normalize(mat3(normal_matrix) * tangent);
		vec3 B = normalize(mat3(normal_matrix) * bitangent);
		TBN = mat3(T, B, surface_normal);
	}
}
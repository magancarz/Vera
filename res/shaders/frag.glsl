#version 330 core

in vec2 pass_texture_coords;
in vec3 surface_normal;
in vec3 to_light_vector[4];

layout (location = 0) out vec4 out_Color;

uniform sampler2D texture_sampler;

uniform vec3 light_color[4];

void main(void) {
	vec3 unit_normal = normalize(surface_normal);
	vec3 total_diffuse = vec3(0.0);

	for(int i = 0; i < 4; i++) {
		vec3 unit_light_vector = normalize(to_light_vector[i]);
		float n_dot1 = dot(unit_normal, unit_light_vector);
		float brightness = max(n_dot1, 0.0);
		total_diffuse = total_diffuse + (brightness * light_color[i]);
	}

	total_diffuse = max(total_diffuse, 0.0);

	vec4 texture_color = texture(texture_sampler, pass_texture_coords);
	out_Color = vec4(total_diffuse, 1.0) * texture_color;
}
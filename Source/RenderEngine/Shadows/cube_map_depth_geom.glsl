#version 330 core

layout (triangles) in;
layout (triangle_strip, max_vertices=18) out;

uniform mat4 light_view_transforms[6];

out vec4 fragment_position;

void main()
{
    for (int face = 0; face < 6; ++face)
    {
        gl_Layer = face;
        for (int i = 0; i < 3; ++i)
        {
            fragment_position = gl_in[i].gl_Position;
            gl_Position = light_view_transforms[face] * fragment_position;
            EmitVertex();
        }

        EndPrimitive();
    }
}

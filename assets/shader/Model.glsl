
#type vertex
#version 460 core
layout(location = 0) in vec3 a_Position;
uniform mat4 u_ViewProjection;
uniform mat4 u_Transform;
out vec4 v_Position;

void main()
{
    gl_Position = u_ViewProjection*u_Transform*vec4(a_Position, 1.0f);
    v_Position = gl_Position;
}

#type fragment
#version 460 core
out vec4 o_Color;
in vec4 v_Position;
void main()
{
   o_Color = v_Position;
}

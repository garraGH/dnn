#type vertex
#version 460 core
layout(location = 0) in vec3 a_Position;
uniform mat4 u_World2Clip;
uniform mat4 u_Model2World;

void main()
{
    gl_Position = u_World2Clip*u_Model2World*vec4(a_Position, 1.0f);
}

#type fragment
#version 460 core
out vec4 o_Color;
void main()
{
    o_Color = vec4(0.8, 0.3, 0.1, 1.0);
}

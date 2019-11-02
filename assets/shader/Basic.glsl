#type vertex
#version 460 core
layout(location = 0) in vec3 a_Position;
layout(location = 1) in vec4 a_Color;
uniform mat4 u_World2Clip;
uniform mat4 u_Model2World;
out vec4 v_Position;
out vec4 v_Color;

void main()
{
    gl_Position = u_World2Clip*u_Model2World*vec4(a_Position, 1.0f);
    v_Position = gl_Position;
    v_Color = a_Color;
}

#type fragment
#version 460 core
in vec4 v_Position;
out vec4 o_Color;
in vec4 v_Color;
uniform vec4 u_Color;
void main()
{
//    o_Color = v_Position;
//    o_Color = v_Color;
    o_Color = u_Color;
}

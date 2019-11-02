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
uniform vec3 u_ModelColor = vec3(1, 0, 0);
uniform vec3 u_LightColor = vec3(0.8, 0.8, 0.8);

out vec4 o_Color;

void main()
{
   o_Color = vec4(u_ModelColor*u_LightColor, 1.0);
}

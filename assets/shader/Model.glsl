
#type vertex
#version 460 core
layout(location = 0) in vec3 a_Position;
uniform mat4 u_World2Clip;
uniform mat4 u_Model2World;
out vec4 v_Position;

void main()
{
    gl_Position = u_World2Clip*u_Model2World*vec4(a_Position, 1.0f);
    v_Position = gl_Position/gl_Position.w;
    v_Position = (v_Position+1)*0.5;
}

#type fragment
#version 460 core
in vec4 v_Position;
out vec4 f_Color;

void main()
{
   f_Color = v_Position;
}

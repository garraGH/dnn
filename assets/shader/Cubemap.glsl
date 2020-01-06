#type Vertex
#version 460 core
#line 1

in vec3 a_PositionMS;

layout(std140) uniform Transform
{
    mat4 WS2CS;
}
u_Transform;

uniform mat4 u_MS2WS;
out vec3 v_LocalPos;

void main()
{
    v_LocalPos = vec3(a_PositionMS.x, -a_PositionMS.y, a_PositionMS.z);
    gl_Position = u_Transform.WS2CS*u_MS2WS*vec4(a_PositionMS, 1.0);
}


#type Fragment
#version 460 core
#line 1

in vec3 v_LocalPos;
out vec4 f_Color;
uniform samplerCube u_Cubemap;

void main()
{
    vec3 color = texture(u_Cubemap, v_LocalPos).rgb;
    f_Color = vec4(color, 1.0);
}

#type Vertex
#version 460 core
#line 1

in vec3 a_Position;
in vec3 a_TexCoord;

layout(std140) uniform Transform
{
    mat4 WS2CS;
}
u_Transform;

uniform mat4 u_MS2WS;
out vec3 v_TexCoord;

void main()
{
    v_TexCoord = a_TexCoord;
    gl_Position = u_Transform.WS2CS*vec4(a_Position, 1.0);
}


#type Fragment
#version 460 core
#line 1

in vec3 v_TexCoord;
out vec4 f_Color;
uniform samplerCube u_Cubemap;
uniform int u_LOD = 0;

void main()
{
    vec3 color = textureLod(u_Cubemap, v_TexCoord, u_LOD).rgb;
    f_Color = vec4(color, 1.0);
}

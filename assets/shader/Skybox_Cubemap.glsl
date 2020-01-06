
#type vertex
#version 460 core

in vec3 a_Position;
out vec3 v_TexCoord;

uniform mat4 u_WS2VS;
uniform mat4 u_VS2CS;

void main()
{
    v_TexCoord = a_Position;
    vec4 pos = u_VS2CS*u_WS2VS*vec4(a_Position, 1.0);
    gl_Position = pos.xyww;
}

#type fragment
#version 460 core

uniform samplerCube u_Skybox;
uniform int u_LOD = 0;
uniform float u_Gamma = 2.2f;
in vec3 v_TexCoord;
out vec4 f_Color;
out vec4 f_ColorBright;


void main()
{
    f_Color = textureLod(u_Skybox, v_TexCoord, u_LOD);
    f_Color = pow(f_Color, vec4(u_Gamma));
    f_ColorBright = vec4(0.0);
}

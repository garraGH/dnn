
#type vertex
#version 460 core
#line 1

in vec3 a_Position;
in vec2 a_TexCoord;

uniform mat4 u_MS2WS;
layout(std140) uniform Transform
{
    mat4 WS2CS;
}
u_Transform;

out vec2 v_TexCoord;

void main()
{
    v_TexCoord = a_TexCoord;
    gl_Position = u_Transform.WS2CS*u_MS2WS*vec4(a_Position, 1.0);
}

#type fragment
#version 460 core
#line 1

in vec2 v_TexCoord;
uniform sampler2D u_Texture;
out vec4 f_Color;

void main()
{
    f_Color = vec4(texture(u_Texture, v_TexCoord).rg, 0, 1);
}


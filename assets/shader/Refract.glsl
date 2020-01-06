#type vertex
#version 460 core

in vec3 a_Position;
in vec3 a_Normal;

out vec3 v_FragPos;
out vec3 v_Normal;

uniform mat4 u_MS2WS;
layout(std140) uniform Transform
{
    mat4 WS2CS;
}
u_Transform;

void main()
{
    v_Normal = mat3(transpose(inverse(u_MS2WS)))*a_Normal;
    vec4 pos = u_MS2WS*vec4(a_Position, 1);
    v_FragPos = pos.xyz;
    gl_Position = u_Transform.WS2CS*pos;
}

#type fragment
#version 460 core

struct Camera
{
    vec3 Position;
};

uniform Camera u_Camera;

uniform float u_RefractiveIndex;
uniform samplerCube u_Skybox;
in vec3 v_Normal;
in vec3 v_FragPos;

out vec4 f_Color;


void main()
{
    vec3 I = normalize(v_FragPos-u_Camera.Position);
    vec3 R = refract(I, normalize(v_Normal), 1.0/u_RefractiveIndex);
    f_Color = vec4(texture(u_Skybox, R).rgb, 1.0);
}


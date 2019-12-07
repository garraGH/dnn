#type vertex
#version 460 core
#line 1

in vec3 a_PositionMS;

#ifdef INSTANCE
in mat4 a_MS2WS;
in vec3 a_Color;
in float a_Intensity;
out VS_OUT
{
    vec3 Color;
    float Intensity;
}
v_Out;
#else
uniform mat4 u_MS2WS = mat4(1);
#endif

struct Camera
{
    vec3 PositionWS;
};


// uniform buffer
layout(std140) uniform Transform
{
    mat4 WS2CS;
}
u_Transform;


void main()
{
#ifndef INSTANCE
    gl_Position = u_Transform.WS2CS*u_MS2WS*vec4(a_PositionMS, 1);
#else
    gl_Position = u_Transform.WS2CS*a_MS2WS*vec4(a_PositionMS, 1);
    v_Out.Color = a_Color;
    v_Out.Intensity = a_Intensity;
#endif
}


//---------------------------------------------------------------------------------
#type fragment
#version 460 core
#line 1

#ifdef INSTANCE
in VS_OUT
{
    vec3 Color;
    float Intensity;
}
f_In;
#else
uniform vec3 u_Color = vec3(1, 0, 0);
uniform float u_Intensity = 1.0f;
#endif

uniform float u_BloomThreshold = 1.0f;

layout(location = 0) out vec4 f_Color;
layout(location = 1) out vec4 f_ColorBright;

void main()
{
    vec3 color;
#ifndef INSTANCE
    color = u_Color*u_Intensity;
#else
    color = f_In.Color*f_In.Intensity;
#endif

    f_Color = vec4(color, 1);

    float brightness = dot(color, vec3(0.2126, 0.7152, 0.0722));
    f_ColorBright = vec4(step(u_BloomThreshold, brightness)*color, 1.0);
}

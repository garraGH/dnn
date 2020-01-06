
#type vertex
#version 460 core
#line 1

in vec3 a_Position;
out vec3 v_Position;

#ifdef SEPERATEVP
uniform mat4 u_WS2VS;
uniform mat4 u_VS2CS;
#else
layout(std140) uniform Transform
{
    mat4 WS2CS;
}
u_Transform;
#endif

void main()
{
    v_Position = a_Position;
#ifdef SEPERATEVP
    gl_Position = u_VS2CS*u_WS2VS*vec4(a_Position, 1);
#else
    gl_Position = u_Transform.WS2CS*vec4(a_Position, 1);
#endif
}

#type fragment
#version 460 core
#line 1

in vec3 v_Position;
out vec4 f_Color;

uniform sampler2D u_EquirectangularMap;
const float PI = 3.14159265;
const vec2 invAtan = vec2(1/(2*PI), -1/PI);

vec2 _SampleSphericalMap(vec3 v)
{
    vec2 uv = vec2(atan(v.z, v.x), asin(v.y));
    uv *= invAtan;
    uv += 0.5;
    return uv;
}

void main()
{
    vec2 uv = _SampleSphericalMap(normalize(v_Position));
    vec3 color = texture(u_EquirectangularMap, uv).rgb;
    f_Color = vec4(color, 1.0);
}


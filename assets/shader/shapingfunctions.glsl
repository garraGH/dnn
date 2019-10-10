#type vertex
#version 460 core
layout(location = 0) in vec3 a_Position;
uniform mat4 u_ViewProjection;
uniform mat4 u_Transform;

void main()
{
    gl_Position = u_ViewProjection*u_Transform*vec4(a_Position, 1.0f);
}

#type fragment
#version 460 core
uniform vec4 u_Color;
uniform vec2 u_Resolution;
uniform float u_Time;

out vec4 color;

vec4 _smoothstep()
{
    vec2 nc = gl_FragCoord.xy/u_Resolution;
    float pct = smoothstep(nc.x-0.01, nc.x, nc.y) - smoothstep(nc.x, nc.x+0.01, nc.y);
    return (1-pct)*vec4(nc.x)+pct*vec4(0.1, 0.8, 0.2, 1.0);
}

void main()
{
    color = _smoothstep();
}

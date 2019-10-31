
#type vertex
#version 460 core

in vec2 a_Position;

uniform vec3 u_NearCorners[4];
uniform vec3 u_FarCorners[4];

out vec3 v_Near;
out vec3 v_Far;

void main()
{
    gl_Position = vec4(a_Position, 0.999, 1);
    v_Near = u_NearCorners[gl_VertexID];
    v_Far = u_FarCorners[gl_VertexID];
}


#type fragment
#version 460 core
#define PI 3.14159265
#define TWO_PI 6.28318530

in vec3 v_Near;
in vec3 v_Far;
out vec4 o_Color;
uniform mat4 u_ViewProjection;
float CheckerBoard(vec2 I, float s)
{
    return float((int(floor(I.x*s))+int(floor(I.y*s)))%2);
}

float ComputeDepth(vec3 pos_ws)
{
    vec4 pos_cs = u_ViewProjection*vec4(pos_ws, 1.0);
    float depth_cs = pos_cs.z/pos_cs.w;
    float far = gl_DepthRange.far;
    float near = gl_DepthRange.near;

    float depth = (((far-near)*depth_cs)+near+far)/2.0;
    return depth;
}

void main()
{
    float t = -v_Near.y/(v_Far.y-v_Near.y);
    vec3 I = v_Near+t*(v_Far-v_Near);
    float c = CheckerBoard(I.xz, 1)*0.3+CheckerBoard(I.xz, 10)*0.2+CheckerBoard(I.xz, 100)*0.1+0.1;
    float spotlight = min(1.0, 1.5-0.02*length(I.xz));
    c *= spotlight*float(t>=0);
    o_Color = vec4(vec3(c), 1);
    gl_FragDepth = ComputeDepth(I);
}

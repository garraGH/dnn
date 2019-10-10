
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
uniform vec2 u_Resolution;
uniform vec4 u_Color;
out vec4 color;

vec4 rect_inner(const vec2 lb, const vec2 rt, float n, vec4 color)
{
    float width_half = n/2/u_Resolution.x;
    vec2 nc = gl_FragCoord.xy/u_Resolution.xy;
    float left   =   smoothstep(lb.x-width_half, lb.x+width_half, nc.x);
    float bottom =   smoothstep(lb.y-width_half, lb.y+width_half, nc.y);
    float right  = 1-smoothstep(rt.x-width_half, rt.x+width_half, nc.x);
    float top    = 1-smoothstep(rt.y-width_half, rt.y+width_half, nc.y);
    return left*bottom*right*top*color;
}

vec4 rect_outter(const vec2 lb, const vec2 rt, float n, vec4 color)
{
    float width_half = n/2/u_Resolution.x;
    vec2 nc = gl_FragCoord.xy/u_Resolution.xy;
    float left   =   smoothstep(lb.x-width_half, lb.x+width_half, nc.x);
    float bottom =   smoothstep(lb.y-width_half, lb.y+width_half, nc.y);
    float right  = 1-smoothstep(rt.x-width_half, rt.x+width_half, nc.x);
    float top    = 1-smoothstep(rt.y-width_half, rt.y+width_half, nc.y);
    return (1-left*bottom*right*top)*color;
}

vec4 rect_filled(const vec2 lb, const vec2 rt, float n, vec4 color)
{
    return rect_inner(lb, rt, n, color);
}

// vec4 rect_outline(const vec2 lb, const vec2 rt, float n, vec4 color)
// {
//     float width_quarter= 0.25*n/u_Resolution.x;
//     return rect_inner(lb-vec2(width_quarter), rt+vec2(width_quarter), 1, color)*rect_outter(lb+vec2(width_quarter), rt-vec2(width_quarter), 1, color);
// }
// 
vec4 rect_outline(const vec2 lb, const vec2 rt, float n, vec4 color)
{
    float width_quarter= 0.25*n/u_Resolution.x;
    return rect_inner(lb-vec2(width_quarter), rt+vec2(width_quarter), 1, color)-rect_inner(lb+vec2(width_quarter), rt-vec2(width_quarter), 1, color);
}


vec4 circle_inner(const vec2 center, float radius, float n, vec4 color)
{
    float width_half = n/2/u_Resolution.x; 
    vec2 nc = gl_FragCoord.xy/u_Resolution.xy;
    float l = distance(nc, center);
    return (1-smoothstep(radius-width_half, radius+width_half, l))*color;
}

vec4 circle_outter(const vec2 center, float radius, float n, vec4 color)
{
    float width_half = n/2/u_Resolution.x; 
    vec2 nc = gl_FragCoord.xy/u_Resolution.xy;
    float l = distance(nc, center);
    return smoothstep(radius-width_half, radius+width_half, l)*color;
}

vec4 circle_filled(const vec2 center, float radius, float n, vec4 color)
{
    return circle_inner(center, radius, n, color);
}

vec4 circle_outline(const vec2 center, float radius, float n, vec4 color)
{
    float width_quarter= 0.25*n/u_Resolution.x; 
    return circle_inner(center, radius+width_quarter, 2, color)-circle_inner(center, radius-width_quarter, 2, color);
}

// vec4 circle_outline(const vec2 center, float radius, float n, vec4 color)
// {
//     float width_quarter= 0.25*n/u_Resolution.x; 
//     return circle_inner(center, radius+width_quarter, 2, color)*circle_outter(center, radius-width_quarter, 2, color);
// }
// 
void main()
{
    vec4 red = vec4(1.0, 0.1, 0.2, 1.0);
    vec4 green = vec4(0.1, 1.0, 0.2, 1.0);
    vec4 blue = vec4(0.1, 0.2, 1.0, 1.0);
    color = rect_outline(vec2(0.1, 0.2), vec2(0.2, 0.3), 1, red) 
          + rect_filled(vec2(0.3, 0.2), vec2(0.4, 0.4), 1, green)
          + circle_filled(vec2(0.5, 0.7), 0.1, 1, red)
          + circle_outline(vec2(0.3, 0.8), 0.1, 4, blue);
}


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
uniform int u_Style;
uniform int u_Mode;
uniform float u_LineWidth = 2;
uniform vec4 u_Colors[2];
uniform vec2 u_Points[3];
out vec4 color;

float rect_inner(const vec2 lb, const vec2 rt, float n)
{
    float width_half = n/2/u_Resolution.x;
    vec2 nc = gl_FragCoord.xy/u_Resolution.xy;
    float left   =   smoothstep(lb.x-width_half, lb.x+width_half, nc.x);
    float bottom =   smoothstep(lb.y-width_half, lb.y+width_half, nc.y);
    float right  = 1-smoothstep(rt.x-width_half, rt.x+width_half, nc.x);
    float top    = 1-smoothstep(rt.y-width_half, rt.y+width_half, nc.y);
    return left*bottom*right*top;
}

float rect_outter(const vec2 lb, const vec2 rt, float n)
{
    return 1-rect_inner(lb, rt, n);
// 
//     float width_half = n/2/u_Resolution.x;
//     vec2 nc = gl_FragCoord.xy/u_Resolution.xy;
//     float left   =   smoothstep(lb.x-width_half, lb.x+width_half, nc.x);
//     float bottom =   smoothstep(lb.y-width_half, lb.y+width_half, nc.y);
//     float right  = 1-smoothstep(rt.x-width_half, rt.x+width_half, nc.x);
//     float top    = 1-smoothstep(rt.y-width_half, rt.y+width_half, nc.y);
//     return 1-left*bottom*right*top;
}

float rect_filled(const vec2 lb, const vec2 rt, float n)
{
    return rect_inner(lb, rt, n);
}

// float rect_outline(const vec2 lb, const vec2 rt, float n)
// {
//     float width_quarter= 0.25*n/u_Resolution.x;
//     return rect_inner(lb-vec2(width_quarter), rt+vec2(width_quarter), 1, color)*rect_outter(lb+vec2(width_quarter), rt-vec2(width_quarter), 1);
// }
// 
float rect_outline(const vec2 lb, const vec2 rt, float n)
{
    float width_quarter= 0.25*n/u_Resolution.x;
    return rect_inner(lb-vec2(width_quarter), rt+vec2(width_quarter), 1)-rect_inner(lb+vec2(width_quarter), rt-vec2(width_quarter), 1);
}


float circle_inner(const vec2 center, float radius, float n)
{
    float width_half = n/2/u_Resolution.x; 
    vec2 nc = gl_FragCoord.xy/u_Resolution.xy;
    float l = distance(nc, center);
    return 1-smoothstep(radius-width_half, radius+width_half, l);
}

float circle_outter(const vec2 center, float radius, float n)
{
    float width_half = n/2/u_Resolution.x; 
    vec2 nc = gl_FragCoord.xy/u_Resolution.xy;
    float l = distance(nc, center);
    return smoothstep(radius-width_half, radius+width_half, l);
}

float circle_filled(const vec2 center, float radius, float n)
{
    return circle_inner(center, radius, n);
}

float circle_outline(const vec2 center, float radius, float n)
{
    float width_quarter= 0.25*n/u_Resolution.x; 
    return circle_inner(center, radius+width_quarter, 2)-circle_inner(center, radius-width_quarter, 2);
}

// float circle_outline(const vec2 center, float radius, float n)
// {
//     float width_quarter= 0.25*n/u_Resolution.x; 
//     return circle_inner(center, radius+width_quarter, 2)*circle_outter(center, radius-width_quarter, 2);
// }
// 
//

float line(const vec2 beg, const vec2 end, float n)
{
    vec2 pixel = gl_FragCoord.xy/u_Resolution;
    float width = 0.25*n/u_Resolution.x;

    float a = distance(pixel, u_Points[0]);
    float b = distance(pixel, u_Points[1]);
    float c = distance(u_Points[0], u_Points[1]);

    if(a>=c || b>=c)
    {
        return 0;
    }

    float p = (a+b+c)*0.5;
    float h = 2.0/c*sqrt(p*(p-a)*(p-b)*(p-c));
    return mix(1.0, 0.0, smoothstep(0.5*width, 1.5*width, h));
}

float sdf_torus(vec2 pixel, vec2 center, float radius, float width)
{
    float d = distance(pixel, center);
    return abs(d-radius)-width;
}

float sdf_circle(vec2 pixel, vec2 center, float radius)
{
    return distance(pixel, center)-radius;
}

float sdf_elipse(vec2 pixel, vec2 center, float a, float b)
{
    float aa = a*a;
    float bb = b*b;
    vec2 offset = pixel-center;
    float xx = offset.x*offset.x;
    float yy = offset.y*offset.y;
    return (bb*xx+aa*yy)/(aa*bb)-1.0;
}

float sdf_box(vec2 pixel, vec2 center, float width, float height)
{
    vec2 d = abs(pixel-center)-vec2(width, height);
    return min(max(d.x, d.y), 0.0)+length(max(d, 0.0));
}

float sdf_roundbox(vec2 pixel, vec2 center, float width, float height, float radius)
{
    return sdf_box(pixel, center, width, height)-radius;
}


vec4 render_fill(float d, vec4 c, float width)
{
    float w = width*0.5/u_Resolution.x;
    return c*(1-smoothstep(-w, w, d));
}

vec4 render_line(float d, vec4 c, float width)
{
    float w = width*0.5/u_Resolution.x;
    return c*(1-smoothstep(-w, w, abs(d)));
}

vec4 render_both(float d, vec4 cf, vec4 cl, float width)
{
    vec4 cFill = render_fill(d, cf, width);
    vec4 cLine = render_line(d, cl, width);

    return mix(cFill, cLine, step(0.01, cLine));
}

vec4 Line()
{
    return line(u_Points[0], u_Points[1], u_LineWidth)*u_Colors[1];
}

void Rectangle()
{
    vec2 lb = min(u_Points[0], u_Points[1]);
    vec2 rt = max(u_Points[0], u_Points[1]);
    switch(u_Mode)
    {
        case 0: 
            color = rect_filled(lb, rt, u_LineWidth)*u_Colors[0];
            break;
        case 1:
            color = rect_outline(lb, rt, u_LineWidth)*u_Colors[1];
            break;
        case 2: 
        {
            float cFill = rect_filled(lb, rt, u_LineWidth);
            float cLine = rect_outline(lb, rt, u_LineWidth);
            color = mix(cFill*u_Colors[0], cLine*u_Colors[1], step(0.1, cLine));
            break;
        }
    }
}

vec4 render(float d)
{
    switch(u_Mode)
    {
        case 0: return render_fill(d, u_Colors[0], u_LineWidth);
        case 1: return render_line(d, u_Colors[1], u_LineWidth);
        case 2: return render_both(d, u_Colors[0], u_Colors[1], u_LineWidth);
    }
}

vec4 Circle()
{
    vec2 pixel = gl_FragCoord.xy/u_Resolution;
    float radius = distance(u_Points[1], u_Points[0]);
    float d = sdf_circle(pixel, u_Points[0], radius);
    return render(d);
}

vec4 Box()
{
    vec2 pixel = gl_FragCoord.xy/u_Resolution;
    vec2 center = (u_Points[0]+u_Points[1])*0.5;
    vec2 size = abs(u_Points[0]-u_Points[1])/2;
    float d = sdf_box(pixel, center, size.x, size.y);
    return render(d);
}

// vec4 RoundBox()
// {
//     vec2 pixel = gl_FragCoord.xy/u_Resolution;
//     vec2 center = (u_Points[0]+u_Points[1])*0.5;
//     vec2 size = abs(u_Points[0]-u_Points[1])/2;
//     float d = sdf_roundbox(pixel, center, size.x, size.y, size.x/10);
//     return render(d);
// }

vec4 Elipse()
{
    vec2 pixel = gl_FragCoord.xy/u_Resolution;
    vec2 center = u_Points[0];
    float a = abs(u_Points[1].x-center.x);
    float b = abs(u_Points[1].y-center.y);
    float d = sdf_elipse(pixel, center, a, b);
    return render(d);
}

vec4 Torus()
{
    vec2 pixel = gl_FragCoord.xy/u_Resolution;
    vec2 center = u_Points[0];
    float radius = distance(u_Points[1], center);
    float width = radius*0.2;
    float d = sdf_torus(pixel, center, radius, width);
    return render(d);
}

void Ring()
{
    color = u_Colors[0];
}

void Default()
{
//     color = vec4(1);
    color = u_Colors[0];
}

void main()
{
    switch(u_Style)
    {
        case 0: color = Line();         break;
        case 1: color = Box();          break;
        case 2: color = Circle();       break;
        case 3: color = Elipse();       break;
        case 4: color = Torus();        break;
        default: Default();     
    }
}

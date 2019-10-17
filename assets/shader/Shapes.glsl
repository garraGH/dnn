
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
uniform int u_Style = 1;
uniform int u_Mode = 0;
uniform float u_LineWidth = 2;
uniform vec4 u_Colors[2];
uniform vec2 u_Points[2];
uniform float u_TorusWidth = 0.1;
uniform float u_Number = 3;
uniform float u_RoundRadius = 0.1;
uniform float u_SawTooth = 0.5;


out vec4 color;

#define PI 3.14159265
#define TWO_PI 6.28318530

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

float sdf_segment(vec2 pixel, const vec2 beg, const vec2 end, float n)
{
    float width = 0.25*n/u_Resolution.x;

    float a = distance(pixel, u_Points[0]);
    float b = distance(pixel, u_Points[1]);
    float c = distance(u_Points[0], u_Points[1]);

    if(a>=c || b>=c)
    {
        return 1;
    }

    float p = (a+b+c)*0.5;
    float h = 2.0/c*sqrt(p*(p-a)*(p-b)*(p-c));
    return mix(0.0, 1.0, smoothstep(0.5*width, 1.5*width, h));
}

float sdf_line(vec2 pixel, vec2 beg, vec2 end, float width)
{
    float a = beg.y-end.y;
    float b = end.x-beg.x;
    return abs(a*pixel.x+b*pixel.y+beg.x*end.y-end.x*beg.y)/sqrt(a*a+b*b);
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

float sdf_polygon(vec2 pixel, vec2 center, float diameter, float N, float sawtooth)
{
    pixel = 2*pixel-1;
    center = 2*center-1;
    pixel -= center;
    float a = atan(pixel.x, pixel.y)+PI;
    float r = TWO_PI/N;
    return cos(floor(sawtooth+a/r)*r-a)*length(pixel)-diameter;
}

float sdf_petal(vec2 pixel, vec2 center, float radius, float N, float sawtooth)
{
    pixel -= center;
    float r = length(pixel);
    float a = atan(pixel.y, pixel.x);
    float f =  radius*(abs(cos(a*N)*sawtooth+(1-sawtooth)));
    return r-f;
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

vec4 Line(const vec2 pixel)
{
    float d = sdf_line(pixel, u_Points[0], u_Points[1], u_LineWidth);
    return render(d);
}

vec4 Segment(const vec2 pixel)
{
    float d = sdf_segment(pixel, u_Points[0], u_Points[1], u_LineWidth);
    return render(d);
}

vec4 Circle(const vec2 pixel)
{
    float radius = distance(u_Points[1], u_Points[0]);
    float d = sdf_circle(pixel, u_Points[0], radius);
    return render(d);
}

vec4 Box(const vec2 pixel)
{
    vec2 center = (u_Points[0]+u_Points[1])*0.5;
    vec2 size = abs(u_Points[0]-u_Points[1])/2;
    float d = sdf_box(pixel, center, size.x, size.y);
    return render(d);
}

vec4 RoundBox(const vec2 pixel)
{
    vec2 center = (u_Points[0]+u_Points[1])*0.5;
    vec2 size = abs(u_Points[0]-u_Points[1])/2;
    float radius = u_RoundRadius*min(size.x, size.y);
    size -= radius;
    float d = sdf_roundbox(pixel, center, size.x, size.y, radius);
    return render(d);
}

vec4 Elipse(const vec2 pixel)
{
    vec2 center = u_Points[0];
    float a = abs(u_Points[1].x-center.x);
    float b = abs(u_Points[1].y-center.y);
    float d = sdf_elipse(pixel, center, a, b);
    return render(d);
}

vec4 Torus(const vec2 pixel)
{
    vec2 center = u_Points[0];
    float radius = distance(u_Points[1], center);
    float d = sdf_torus(pixel, center, radius, u_TorusWidth*radius);
    return render(d);
}

vec4 Polygon(const vec2 pixel)
{
    float diameter = 2.0*distance(u_Points[0], u_Points[1]);
    float d = sdf_polygon(pixel, u_Points[0], diameter, u_Number, u_SawTooth);
    return render(d);
}

vec4 Petal(const vec2 pixel)
{
    float radius = distance(u_Points[0], u_Points[1]);
    float d = sdf_petal(pixel, u_Points[0], radius, u_Number, u_SawTooth);
    return render(d);
}

void Default(const vec2 pixel)
{
    color = u_Colors[0];
}


void main()
{
    vec2 pixel = gl_FragCoord.xy/u_Resolution;

    switch(u_Style)
    {
        case 0: color = Line(pixel);         break;
        case 1: color = Segment(pixel);      break;
        case 2: color = Box(pixel);          break;
        case 3: color = RoundBox(pixel);     break;
        case 4: color = Circle(pixel);       break;
        case 5: color = Elipse(pixel);       break;
        case 6: color = Torus(pixel);        break;
        case 7: color = Polygon(pixel);      break;
        case 8: color = Petal(pixel);      break;

        default: Default(pixel);     
    }
}

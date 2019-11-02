

#type vertex
#version 460 core
layout(location = 0) in vec3 a_Position;
uniform mat4 u_World2Clip;
uniform mat4 u_Model2World;

void main()
{
    gl_Position = u_World2Clip*u_Model2World*vec4(a_Position, 1.0f);
}

#type fragment
#version 460 core

uniform float u_RadRotated;
uniform vec2 u_Resolution;
uniform vec4 u_RadarColor;
uniform vec4 u_CircleColor;
uniform vec2 u_CircleCenter;
uniform float u_CircleRadius;
uniform float u_RadarRange;
uniform float u_Time;
// uniform int u_Direction;

out vec4 color;

#define PI 3.14159265
#define TWO_PI 6.28318530
#define sqrt2 1.414

float sdf_segment(vec2 pixel, const vec2 beg, const vec2 end, float n)
{
    float width = 0.25*n/u_Resolution.x;

    float a = distance(pixel, beg);
    float b = distance(pixel, end);
    float c = distance(beg, end);

    if(a>=c || b>=c)
    {
        return 1;
    }

    float p = (a+b+c)*0.5;
    float h = 2.0/c*sqrt(p*(p-a)*(p-b)*(p-c));
    return mix(0.0, 1.0, smoothstep(0.5*width, 1.5*width, h));
}

float sdf_circle(vec2 pixel, vec2 center, float radius)
{
    return distance(pixel, center)-radius;
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

vec4 render(float d, vec4 c, float width)
{
    float w = width*0.5/u_Resolution.x;
    return c*(1-smoothstep(-w, w, abs(d)));
}

vec4 Segment(const vec2 pixel, const vec2 beg, const vec2 end, vec4 c, float width)
{
    float d = sdf_segment(pixel, beg, end, width);
    return render(d, c, width);
}

vec4 Circle(const vec2 pixel, const vec2 center, float radius, vec4 c, float width)
{
    float d = sdf_circle(pixel, center, radius);
    return render(d, c, width);
}

vec4 Circle(const vec2 pixel, const vec2 center, float radius, vec4 c)
{
    float d = radius-distance(pixel, center);
    return c*smoothstep(-0.001, +0.001, d);
}

float Circle(const vec2 pixel, const vec2 center, float radius)
{
    float d = distance(pixel, center);
    d *= 1-smoothstep(radius-0.002, radius+0.002, d);
    return smoothstep(0.5*radius, radius, d);
}

vec4 Triangle(const vec2 pixel, const vec2 center, float radius, vec4 color)
{
    float d = sdf_polygon(pixel, center, 2*radius, 3, 0.5);
    return (1-step(0, d))*color;
}

float Sector(const vec2 pixel, const vec2 center, float radius, float beg, float range)
{
    vec2 offset = pixel-center;
    float r = length(offset);
    float a = atan(offset.y, offset.x);
    a += PI;
//     float d = mod(range-(a+TWO_PI-beg), TWO_PI); //CW
    float d = mod((a+TWO_PI-beg), TWO_PI); //CCW
    d = max(0, d);
    d *= 1-step(range, d);
    d *= 1-step(radius, r);
    return d;
}

vec4 Arc(const vec2 pixel, const vec2 center, float radius, float beg, float range, vec4 c, float width)
{
    vec2 offset = pixel-center;
    float r = length(offset);
    float a = atan(offset.y, offset.x);
    a += PI;
    float dr = abs(r-radius);
    dr = 1-smoothstep(0.0, width, dr);
    float da = mod(a+TWO_PI-beg, TWO_PI);
    dr *= 1-step(range, da);

    return dr*c;
}

mat2 _rotate2d(float a)
{
    return mat2(+cos(a), -sin(a),
                +sin(a), +cos(a));
}

vec2 rotate(in vec2 pixel, float a)
{
    pixel *= _rotate2d(a);
    return pixel;
}

void main()
{
    vec2 center = vec2(0);
    vec2 pixel = gl_FragCoord.xy/u_Resolution;
    pixel -= 0.5;
    float l = 0.25;
    float radius = l*sqrt2;
    color = Segment(pixel, vec2(-l, -l), vec2(l, l), vec4(0.5), 1);
    color = max(color, Segment(pixel, vec2(-l, l), vec2(l, -l), vec4(0.5), 1)); 
    color = max(color, Circle(pixel, center, radius*0.7, vec4(0.4, 1.0, 0.88, 1.0), 5));
    color = max(color, Circle(pixel, center, radius*0.4, vec4(0.4, 1.0, 0.88, 1.0), 5));
    color = max(color, Circle(pixel, center, radius*0.025, vec4(0.2, 1.0, 0.88, 1.0), 6));
    color = max(color, Circle(pixel, center, radius*0.025, vec4(0.2, 1.0, 0.88, 1.0), 6));
    color = max(color, Circle(pixel, center, radius*0.025, vec4(0.2, 1.0, 0.88, 1.0), 6));
    color = max(color, Arc(pixel, center, 0.48, PI/72.0, PI*35.0/36.0, vec4(0.9, 1.0, 0.9, 1.0), 0.004));
    color = max(color, Arc(pixel, center, 0.48, PI*73.0/72.0, PI*35.0/36.0, vec4(0.9, 1.0, 0.9, 1.0), 0.004));
    float da = abs(sin(u_Time));
    color = max(color, Arc(pixel, center, 1.08*radius, PI/36.0+da, PI*17.0/18.0-2*da, vec4(0.2, 1.0, 0.88, 1.0), 0.001));
    color = max(color, Arc(pixel, center, 1.08*radius, PI*37/36.0+da, PI*17.0/18.0-2*da, vec4(0.2, 1.0, 0.88, 1.0), 0.001));
    float beg = TWO_PI-u_RadarRange;
    beg += u_RadRotated;
    beg = mod(beg, TWO_PI);
    float end = beg+u_RadarRange-PI;
    color = max(color, Segment(pixel, center, radius*vec2(cos(end), sin(end)), u_RadarColor, 8));
    color = max(color, Circle(pixel, center, radius, vec4(1), 10));
    color = max(color, Circle(pixel, u_CircleCenter, 0.01, u_CircleColor, 4));
    color = max(color, Circle(pixel, vec2(sin(u_Time)*radius, 0), 0.005, vec4(1)));
    color = max(color, Circle(pixel, vec2(0, cos(u_Time)*radius), 0.005, vec4(1)));
    color = max(color, Circle(pixel, u_CircleCenter, 0.006, u_CircleColor)*step(0.008, mod(u_Time, 0.016)));
    da = 0.1*abs(sin(u_Time));
    color = max(color, Triangle(rotate(pixel, -PI*0.5)+vec2(0, 0.48-da), vec2(0), 0.01, vec4(0.8, 1, 0.9, 1)));
    color = max(color, Triangle(rotate(pixel, +PI*0.5)+vec2(0, 0.48-da), vec2(0), 0.01, vec4(0.8, 1, 0.9, 1)));
    color = mix(color, u_CircleColor, Circle(pixel, u_CircleCenter, u_CircleRadius));
    color = mix(color, u_RadarColor, 0.2*Sector(pixel, center, radius, beg, u_RadarRange));

}

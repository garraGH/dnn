
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

uniform vec2 u_Resolution = vec2(1000, 1000);
uniform float u_Time = 0;
uniform vec2 u_Tiles = vec2(5, 5);
uniform float u_Radius = 0.1;
uniform vec4 u_Color;

out vec4 color;

#define PI 3.14159265358979323846
#define TWO_PI 6.28318530
#define sqrt2 1.414

float circle(in vec2 pixel, in vec2 center, in float radius)
{
    vec2 offset = pixel-center;
    float l = length(offset);
    return 1.0-smoothstep(radius*0.99, radius*1.01, l);
}

float box(const vec2 pixel, vec2 size, float smoothEdges)
{
    size = vec2(0.5)-size;
    vec2 aa = vec2(smoothEdges*0.5);
    vec2 uv = smoothstep(size, size+aa, pixel)*smoothstep(size, size+aa, vec2(1)-pixel);
    return uv.x*uv.y;
}

mat2 _rotateMatrix2D(float rad)
{
    float c = cos(rad);
    float s = sin(rad);
    return mat2( +c, -s, 
                 +s, +c );
}

vec2 rotate2D(inout vec2 pixel, float rad)
{
    pixel -= 0.5;
    pixel *= _rotateMatrix2D(rad);
    pixel += 0.5;
    return pixel;
}

vec2 tile(inout vec2 pixel, in vec2 tiles)
{
    pixel *= tiles;
    pixel = fract(pixel);
    return pixel;
}

vec2 brick(inout vec2 pixel, in vec2 tiles)
{
    pixel *= tiles;
    pixel.x += step(1, mod(pixel.y, 2))*(sin(u_Time)+0.5);
    pixel = fract(pixel);
    return pixel;
}
void main()
{
    vec2 pixel = gl_FragCoord.xy/u_Resolution;

//     rotate2D(pixel, PI*0.25);
//     tile(pixel, u_Tiles);
    brick(pixel, u_Tiles);

//     color = vec4(vec3(circle(pixel, vec2(0.5, 0.5), u_Radius)), 1);

    float a = box(pixel, vec2(0.49, 0.49), 0.01);
    color = a*u_Color;
//     rotate2D(pixel, PI*0.25);
//     color = max(color, vec4(vec3(box(pixel, vec2(0.1, 0.1), 0.01)), 1));
}

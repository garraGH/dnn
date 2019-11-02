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

uniform int u_Style = 0;
uniform float u_Radius = 0.35;
uniform float u_Time;
uniform float u_Speed = 0.1;
uniform int u_ShowSDF = 0;
uniform vec2 u_Resolution;

out vec4 color;

float box(in vec2 pixel, in vec2 size)
{
    size = vec2(0.5)-size*0.5;
    vec2 uv = smoothstep(size, size+vec2(0.001), pixel);
    uv *= smoothstep(size, size+vec2(0.001), 1-pixel);
    return uv.x*uv.y;
}

float cross(in vec2 pixel, float size)
{
    return box(pixel, vec2(size, size*0.25))+box(pixel, vec2(size*0.25, size));
}

void translate(inout vec2 pixel)
{
    float a = u_Time*u_Speed;
    pixel += u_Radius*vec2(cos(a), sin(a));
}

mat2 _rotate2d(float a)
{
    return mat2(+cos(a), -sin(a),
                +sin(a), +cos(a));
}

#define PI 3.14159265
void rotate(inout vec2 pixel)
{
    pixel -= 0.5;
    pixel *= _rotate2d(sin(u_Time*u_Speed)*PI);
    pixel += 0.5;
}

mat2 _scale2d(vec2 s)
{
    return mat2(1.0/s.x, 0.0, 
                0.0, 1.0/s.y);
}

void scale(inout vec2 pixel)
{
    pixel -= 0.5;
    pixel *= _scale2d(vec2(sin(u_Time*u_Speed)+0.2));
    pixel += 0.5;
}

void main()
{
    vec2 pixel = gl_FragCoord.xy/u_Resolution;
    switch(u_Style)
    {
        case 0: 
            translate(pixel);
            break;
        case 1:
            rotate(pixel);
            break;
        case 2:
            scale(pixel);
            break;
        case 3:
            translate(pixel);
            rotate(pixel);
            break;
        case 4:
            translate(pixel);
            scale(pixel);
            break;
        case 5:
            rotate(pixel);
            scale(pixel);
            break;
        default:
            translate(pixel);
            rotate(pixel);
            scale(pixel);
            break;
    }
    color = (u_ShowSDF == 1)? vec4(pixel.x, pixel.y, 0, 1) : vec4(0);
    color += vec4(cross(pixel, 0.25));
}

#type vertex
#version 460 core
layout(location = 0) in vec3 a_Position;
layout(location = 1) in vec2 a_TexCoord;
uniform mat4 u_ViewProjection;
uniform mat4 u_Transform;

out vec2 v_TexCoord;

void main()
{
    gl_Position = u_ViewProjection*u_Transform*vec4(a_Position, 1.0f);
    v_TexCoord = a_TexCoord;
}

#type fragment
#version 460 core
uniform vec4 u_ColorFst;
uniform vec4 u_ColorSnd;
uniform vec2 u_Resolution;
uniform float u_Time;
uniform float u_Speed;
uniform float u_LineWidth;
uniform int u_EasingFunction = 0;
uniform int u_Test = 0;
uniform int u_NumFlags = 1;
uniform int u_FlagLayout = 1;
uniform int u_Direction = -1;

in vec2 v_TexCoord;
uniform sampler2D u_Texture2D_fst;
uniform sampler2D u_Texture2D_snd;

out vec4 color;

#define PI 3.141592653589793
#define HALF_PI 1.5707963267948966


float linear(float t) {
  return t;
}

float exponentialIn(float t) {
  return t == 0.0 ? t : pow(2.0, 10.0 * (t - 1.0));
}

float exponentialOut(float t) {
  return t == 1.0 ? t : 1.0 - pow(2.0, -10.0 * t);
}

float exponentialInOut(float t) {
  return t == 0.0 || t == 1.0
    ? t
    : t < 0.5
      ? +0.5 * pow(2.0, (20.0 * t) - 10.0)
      : -0.5 * pow(2.0, 10.0 - (t * 20.0)) + 1.0;
}

float sineIn(float t) {
  return sin((t - 1.0) * HALF_PI) + 1.0;
}

float sineOut(float t) {
  return sin(t * HALF_PI);
}

float sineInOut(float t) {
  return -0.5 * (cos(PI * t) - 1.0);
}

float qinticIn(float t) {
  return pow(t, 5.0);
}

float qinticOut(float t) {
  return 1.0 - (pow(t - 1.0, 5.0));
}

float qinticInOut(float t) {
  return t < 0.5
    ? +16.0 * pow(t, 5.0)
    : -0.5 * pow(2.0 * t - 2.0, 5.0) + 1.0;
}

float quarticIn(float t) {
  return pow(t, 4.0);
}

float quarticOut(float t) {
  return pow(t - 1.0, 3.0) * (1.0 - t) + 1.0;
}

float quarticInOut(float t) {
  return t < 0.5
    ? +8.0 * pow(t, 4.0)
    : -8.0 * pow(t - 1.0, 4.0) + 1.0;
}

float quadraticInOut(float t) {
  float p = 2.0 * t * t;
  return t < 0.5 ? p : -p + (4.0 * t) - 1.0;
}

float quadraticIn(float t) {
  return t * t;
}

float quadraticOut(float t) {
  return -t * (t - 2.0);
}

float cubicIn(float t) {
  return t * t * t;
}

float cubicOut(float t) {
  float f = t - 1.0;
  return f * f * f + 1.0;
}

float cubicInOut(float t) {
  return t < 0.5
    ? 4.0 * t * t * t
    : 0.5 * pow(2.0 * t - 2.0, 3.0) + 1.0;
}

float elasticIn(float t) {
  return sin(13.0 * t * HALF_PI) * pow(2.0, 10.0 * (t - 1.0));
}

float elasticOut(float t) {
  return sin(-13.0 * (t + 1.0) * HALF_PI) * pow(2.0, -10.0 * t) + 1.0;
}

float elasticInOut(float t) {
  return t < 0.5
    ? 0.5 * sin(+13.0 * HALF_PI * 2.0 * t) * pow(2.0, 10.0 * (2.0 * t - 1.0))
    : 0.5 * sin(-13.0 * HALF_PI * ((2.0 * t - 1.0) + 1.0)) * pow(2.0, -10.0 * (2.0 * t - 1.0)) + 1.0;
}

float circularIn(float t) {
  return 1.0 - sqrt(1.0 - t * t);
}

float circularOut(float t) {
  return sqrt((2.0 - t) * t);
}

float circularInOut(float t) {
  return t < 0.5
    ? 0.5 * (1.0 - sqrt(1.0 - 4.0 * t * t))
    : 0.5 * (sqrt((3.0 - 2.0 * t) * (2.0 * t - 1.0)) + 1.0);
}

float bounceOut(float t) {
  const float a = 4.0 / 11.0;
  const float b = 8.0 / 11.0;
  const float c = 9.0 / 10.0;

  const float ca = 4356.0 / 361.0;
  const float cb = 35442.0 / 1805.0;
  const float cc = 16061.0 / 1805.0;

  float t2 = t * t;

  return t < a
    ? 7.5625 * t2
    : t < b
      ? 9.075 * t2 - 9.9 * t + 3.4
      : t < c
        ? ca * t2 - cb * t + cc
        : 10.8 * t * t - 20.52 * t + 10.72;
}

float bounceIn(float t) {
  return 1.0 - bounceOut(1.0 - t);
}

float bounceInOut(float t) {
  return t < 0.5
    ? 0.5 * (1.0 - bounceOut(1.0 - t * 2.0))
    : 0.5 * bounceOut(t * 2.0 - 1.0) + 0.5;
}

float backIn(float t) {
  return pow(t, 3.0) - t * sin(t * PI);
}

float backOut(float t) {
  float f = 1.0 - t;
  return 1.0 - (pow(f, 3.0) - f * sin(f * PI));
}

float backInOut(float t) {
  float f = t < 0.5
    ? 2.0 * t
    : 1.0 - (2.0 * t - 1.0);

  float g = pow(f, 3.0) - f * sin(f * PI);

  return t < 0.5
    ? 0.5 * g
    : 0.5 * (1.0 - g) + 0.5;
}

float plot(vec2 st, float pct, float width)
{
    width /= u_Resolution.x;
    return smoothstep(pct-width/2, pct, st.y) - smoothstep(pct, pct+width/2, st.y);
}

float _GetPercentage(float t)
{
    switch(u_EasingFunction)
    {
        case 1:  return exponentialIn(t); 
        case 2:  return exponentialOut(t); 
        case 3:  return exponentialInOut(t); 
        case 4:  return sineIn(t); 
        case 5:  return sineOut(t); 
        case 6:  return sineInOut(t); 
        case 7:  return qinticIn(t); 
        case 8:  return qinticOut(t); 
        case 9:  return qinticInOut(t); 
        case 10: return quarticIn(t); 
        case 11: return quarticOut(t); 
        case 12: return quarticInOut(t); 
        case 13: return quadraticIn(t); 
        case 14: return quadraticOut(t); 
        case 15: return quadraticInOut(t); 
        case 16: return cubicIn(t); 
        case 17: return cubicOut(t); 
        case 18: return cubicInOut(t); 
        case 19: return elasticIn(t); 
        case 20: return elasticOut(t); 
        case 21: return elasticInOut(t); 
        case 22: return circularIn(t); 
        case 23: return circularOut(t); 
        case 24: return circularInOut(t); 
        case 25: return bounceIn(t); 
        case 26: return bounceOut(t); 
        case 27: return bounceInOut(t); 
        case 28: return backIn(t); 
        case 29: return backOut(t); 
        case 30: return backInOut(t); 
        default: return linear(t);
    }
}

vec3 RGB2HSB(in vec3 c)
{
    vec4 K = vec4(0.0, -1.0/3.0, 2.0/3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
    float d = q.x-min(q.w, q.y);
    float e = 1e-10;
    return vec3(abs(q.z+(q.w-q.y)/(6.0*d+e)), d/(q.x+e), q.x);
}

vec3 HSB2RGB(in vec3 c)
{
    vec3 rgb = clamp(abs(mod(c.x*6.0+vec3(0.0, 4.0, 2.0), 6.0)-3.0)-1.0, 0.0, 1.0);
    rgb = rgb*rgb*(3.0-2.0*rgb);
    return c.z*mix(vec3(1.0), rgb, c.y);
}

void _ColorTransition(float t)
{
    color = mix(u_ColorFst, u_ColorSnd, _GetPercentage(t));
}

void _TextureTransition(float t)
{
    vec4 color_fst = texture(u_Texture2D_fst, v_TexCoord);
    vec4 color_snd = texture(u_Texture2D_snd, v_TexCoord);
    color = mix(color_fst, color_snd, _GetPercentage(t));
}

void _TextureMixEachChannel()
{
    vec2 st = gl_FragCoord.xy/u_Resolution;
    vec4 pcts = vec4(1.0f);
    pcts.r = linear(st.x);
    pcts.g = sineInOut(st.x);
    pcts.b = circularInOut(st.x);
    vec4 color_fst = texture(u_Texture2D_fst, v_TexCoord);
    vec4 color_snd = texture(u_Texture2D_snd, v_TexCoord);
    color = mix(color_fst, color_snd, pcts);
    const vec4 red   = vec4(1, 0, 0, 1);
    const vec4 green = vec4(0, 1, 0, 1);
    const vec4 blue  = vec4(0, 0, 1, 1);
    color = mix(color, red,   plot(st, pcts.r, u_LineWidth));
    color = mix(color, green, plot(st, pcts.g, u_LineWidth));
    color = mix(color, blue,  plot(st, pcts.b, u_LineWidth));
}

void _ColorFlags_Vertical()
{
    vec2 st = gl_FragCoord.xy/u_Resolution;
    float flagWidth = 1.0/u_NumFlags;
    float currWidth = 0;
    float distance = st.x;
    distance += u_Direction*(u_Time*u_Speed);
    distance = fract(distance);
//     distance = exponentialIn(distance);
    for(int i=0; i<u_NumFlags; i++)
    {
        color.rgb += step(currWidth, distance)*(1-step(currWidth+flagWidth, distance))*HSB2RGB(vec3(currWidth, 1, 1));
        currWidth += flagWidth;
    }
}

void _ColorFlags_Horizontal()
{
    vec2 st = gl_FragCoord.xy/u_Resolution;
    float flagWidth = 1.0/u_NumFlags;
    float currWidth = 0;
    float distance = st.y;
    distance += u_Direction*(u_Time*u_Speed);
    distance = fract(distance);
    for(int i=0; i<u_NumFlags; i++)
    {
        color.rgb += step(currWidth, distance)*(1-step(currWidth+flagWidth, distance))*HSB2RGB(vec3(currWidth, 1, 1));
        currWidth += flagWidth;
    } 
}

#define TWO_PI 6.28381530718
void _ColorFlags_Circle()
{
    vec2 st = gl_FragCoord.xy/u_Resolution;
    vec2 toCenter = st-vec2(0.5);
    float angle = atan(toCenter.y, toCenter.x);
    angle += (1-step(0, angle))*TWO_PI;
    angle /= TWO_PI;
    angle += u_Direction*(u_Time*u_Speed);
    angle = fract(angle);
    float radius = length(toCenter)*2.0;
    float flagAngle = 1.0/u_NumFlags;
    float currAngle = 0;
    for(int i=0; i<u_NumFlags; i++)
    {
        color.rgb += step(currAngle, angle)*(1-step(currAngle+flagAngle, angle))*HSB2RGB(vec3(currAngle, 1, 1));
        currAngle += flagAngle;
    } 
    color *= 1-smoothstep(0.98, 1.02, radius);
}

void _ColorFlags_Rainbow()
{
    vec2 st = gl_FragCoord.xy/u_Resolution;
    vec2 toCenter = st-vec2(0.5);
    
    float radius = length(toCenter)*2;
    if(radius>1)
    {
        color = vec4(0, 0, 0, 1);
        return;
    }
    radius += u_Direction*(u_Time*u_Speed);
    radius = fract(radius);
    float flagWidth = 1.0/u_NumFlags;
    float currWidth = 0;
    for(int i=0; i<u_NumFlags; i++)
    {
        color.rgb += step(currWidth, radius)*(1-step(currWidth+flagWidth, radius))*HSB2RGB(vec3(currWidth, 1, 1));
        currWidth += flagWidth;
    } 
}

void _ColorFlagsAnimation()
{
    switch(u_FlagLayout)
    {
        case 0: _ColorFlags_Vertical();   break;
        case 1: _ColorFlags_Horizontal(); break;
        case 2: _ColorFlags_Circle();     break;
        case 3: _ColorFlags_Rainbow();    break;
    }
}

void _ColorFlagsShrink()
{
    vec2 st = gl_FragCoord.xy/u_Resolution;
    vec2 toCenter = st-vec2(0.5);
    float angle = atan(toCenter.y, toCenter.x);
    angle += (1-step(0, angle))*TWO_PI;
    angle /= TWO_PI;                                                 
    if(angle<1.0/2.0)
    {
        angle /= 1.5;
    }
    else
    {
        angle = 1-(1-angle)/0.75;
    }
    
    float radius = length(toCenter)*2.0;
    color.rgb = HSB2RGB(vec3(angle, radius, 1));
    color *= 1-smoothstep(0.98, 1.02, radius);
}

void _DefaultColor()
{
    color = vec4(0.1, 0.2, 0.3, 1.0);
}

void main()
{
    float t = u_Time*u_Speed;
    t = abs(fract(t)*2.0-1.0);
    vec2 st = gl_FragCoord.xy/u_Resolution;
    switch(u_Test)
    {
        case 0:  _ColorTransition(t);      break;
        case 1:  _TextureTransition(t);    break;
        case 2:  _TextureMixEachChannel(); break;
        case 3:  _ColorFlagsAnimation();   break;
        case 4:  _ColorFlagsShrink();      break;
        default: _DefaultColor();
    }
}

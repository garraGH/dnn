#type vertex
#version 460 core

layout(location = 0) in vec2 a_Position;
uniform vec2 u_LeftBottomTexCoord = vec2(0, 0);
uniform vec2 u_RightTopTexCoord = vec2(1, 1);
out vec2 v_TexCoord;
void main()
{
    gl_Position = vec4(a_Position, 0.0f, 1.0f);
    if(gl_VertexID == 0)
    {
        v_TexCoord = u_LeftBottomTexCoord;
    }
    else if(gl_VertexID == 1)
    {
        v_TexCoord = vec2(u_RightTopTexCoord.x, u_LeftBottomTexCoord.y);
    }
    else if(gl_VertexID == 2)
    {
        v_TexCoord = u_RightTopTexCoord;
    }
    else
    {
        v_TexCoord = vec2(u_LeftBottomTexCoord.x, u_RightTopTexCoord.y);
    }
}

#type fragment
#version 460 core

in vec2 v_TexCoord;
uniform sampler2D u_Offscreen;
uniform float u_Exposure = 0.5f;
uniform float u_Gamma = 2.2f;
uniform int u_PostProcess = 2;// 0: None; 1: Gray; 2: Smooth; 3: Edge
out vec4 f_Color;

void _ToneMap(inout vec3 color)
{
    color = vec3(1.0)-exp(-color*u_Exposure);
}

void _GammaCorrect(inout vec3 color)
{
    color = pow(color, vec3(1.0/u_Gamma));
}

vec4 _Color(vec2 uv)
{
    vec3 color = texture(u_Offscreen, uv).rgb;

#ifdef TONE_MAP
    _ToneMap(color);
#endif

#ifdef GAMMA_CORRECTION
    _GammaCorrect(color);
#endif

    return vec4(color, 1.0);
}

void None()
{
    f_Color = _Color(v_TexCoord);
}

void Gray()
{
    f_Color = _Color(v_TexCoord);
    float gray = 0.2126*f_Color.r+0.7152*f_Color.g+0.0722*f_Color.b;
    f_Color = vec4(vec3(gray), 1.0);
}

#define offset 0.001f
const vec2 offsets[9] = { {-offset, +offset}, 
                          {      0, +offset}, 
                          {+offset, +offset}, 
                          {-offset, 0      },
                          {      0, 0      },
                          {+offset, 0      },
                          {-offset, -offset}, 
                          {      0, -offset}, 
                          {+offset, -offset} };


float kernel_edge[9] = { +1, +1, +1, 
                         +1, -8, +1, 
                         +1, +1, +1 };

float kernel_smooth[9] = { 1.0/16.0, 2.0/16.0, 1.0/16.0, 
                           2.0/16.0, 4.0/16.0, 2.0/16.0, 
                           1.0/16.0, 2.0/16.0, 1.0/16.0 };

void Kernel(float[9] kernel)
{
    vec3 color = vec3(0.0);
    for(int i=0; i<9; i++)
    {
        color += kernel[i]*_Color(v_TexCoord+offsets[i]).rgb;
    }
    f_Color = vec4(color, 1.0);
}

void main()
{
    switch(u_PostProcess)
    {
        case 1:     Gray();                 break;
        case 2:     Kernel(kernel_smooth);  break;
        case 3:     Kernel(kernel_edge);    break;
        default:    None();
    }
}


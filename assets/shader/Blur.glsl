#type vertex
#version 460 core

layout(location = 0) in vec2 a_Position;
uniform vec2 u_LeftBottomTexCoord = {0, 0};
uniform vec2 u_RightTopTexCoord = {1, 1};
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
uniform bool u_Horizontal = true;
uniform float kernel[5] = float[](0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);
uniform sampler2D u_Offscreen;

out vec4 f_Color;

void main()
{
    vec2 tex_offset = 1.0/textureSize(u_Offscreen, 0);
    vec3 result = texture(u_Offscreen, v_TexCoord).rgb*kernel[0];
    if(u_Horizontal)
    {
        for(int i=1; i<5; i++)
        {
            result += texture(u_Offscreen, v_TexCoord+vec2(tex_offset.x*i, 0.0)).rgb*kernel[i];
            result += texture(u_Offscreen, v_TexCoord-vec2(tex_offset.x*i, 0.0)).rgb*kernel[i];
        }
    }
    else
    {
        for(int i=1; i<5; i++)
        {
            result += texture(u_Offscreen, v_TexCoord+vec2(0.0, tex_offset.y*i)).rgb*kernel[i];
            result += texture(u_Offscreen, v_TexCoord-vec2(0.0, tex_offset.y*i)).rgb*kernel[i];
        }
    }
    f_Color = vec4(result, 1.0);
}


#type vertex
#version 460 core

in vec2 a_Position;
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
uniform sampler2D u_Texture;
out vec4 f_Color;

void main()
{
    f_Color = texture(u_Texture, v_TexCoord);
}



#type vertex
#version 460 core

layout(location = 0) in vec2 a_Position;
uniform vec2 u_RightTopTexCoord;
out vec2 v_TexCoord;
void main()
{
    gl_Position = vec4(a_Position, 0.0f, 1.0f);
    v_TexCoord = vec2(gl_VertexID%2, gl_VertexID/2)*u_RightTopTexCoord;
}

#type fragment
#version 460 core

in vec2 v_TexCoord;
uniform sampler2D u_Offscreen;
out vec4 f_Color;
void main()
{
    f_Color = texture(u_Offscreen, v_TexCoord);
    float gray = 0.2126*f_Color.r+0.7152*f_Color.g+0.0722*f_Color.b;
    f_Color = vec4(vec3(gray), 1.0);
}


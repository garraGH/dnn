
#type vertex
#version 460 core

in vec2 a_Position;
out vec3 v_Direction;

uniform vec3 u_CornersDirection[4];
void main()
{
    gl_Position = vec4(a_Position, 1, 1);
    v_Direction = u_CornersDirection[gl_VertexID];
}


#type fragment
#version 460 core
#define PI 3.14159265
#define TWO_PI 6.28318530

uniform sampler2D u_Skybox;
in vec3 v_Direction;
out vec4 o_Color;

void main()
{
    float longitude = acos(v_Direction.y)/PI;
    float latitude = atan(v_Direction.x, -v_Direction.z)/TWO_PI;
    o_Color = texture(u_Skybox, vec2(latitude, longitude));
}

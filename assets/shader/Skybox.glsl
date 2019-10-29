
#type vertex
#version 460 core

in vec2 a_Position;
in vec3 a_Direction;

out vec3 v_Direction;
void main()
{
    gl_Position = vec4(a_Position, 1, 1);
    v_Direction = a_Direction;
}


#type fragment
#version 460 core
#define PI 3.14159265
#define HALF_PI 1.57079632

uniform sampler2D u_Skybox;
uniform vec3 u_CameraDirection;
in vec3 v_Direction;
out vec4 o_Color;

void main()
{
    float longitude = acos(v_Direction.y)/PI;
    float latitude = atan(v_Direction.x, -v_Direction.z)/HALF_PI;
    o_Color = texture(u_Skybox, vec2(latitude, longitude));
}

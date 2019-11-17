
#type vertex
#version 460 core

in vec2 a_Position;
uniform vec3 u_NearCorners[4];
uniform vec3 u_FarCorners[4];
out vec3 v_Near;
out vec3 v_Far;

void main()
{
    gl_Position = vec4(a_Position, 1, 1);
    v_Near = u_NearCorners[gl_VertexID];
    v_Far = u_FarCorners[gl_VertexID];
}


#type fragment
#version 460 core
#define PI 3.14159265
#define TWO_PI 6.28318530

uniform sampler2D u_Skybox;
in vec3 v_Near;
in vec3 v_Far;
out vec4 f_Color;

void main()
{
    vec3 dir = normalize(v_Far-v_Near);
    float latitude = acos(dir.y)/PI;
    float longitude = atan(dir.x, -dir.z)/TWO_PI;
    f_Color = texture(u_Skybox, vec2(longitude, -latitude));
}

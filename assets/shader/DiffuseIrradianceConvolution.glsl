#type vertex
#version 460 core
#line 1

in vec3 a_Position;
out vec3 v_Position;

uniform mat4 u_WS2VS;
uniform mat4 u_VS2CS;

void main()
{
    v_Position = a_Position;
    gl_Position = u_VS2CS*u_WS2VS*vec4(a_Position, 1);
}

#type fragment
#version 460 core
#line 1

in vec3 v_Position;
out vec4 f_Color;

uniform samplerCube u_EnvironmentMap;

const float PI = 3.14159265;

void main()
{
    vec3 N = normalize(v_Position);
    vec3 irradiance = vec3(0);

    vec3 up = vec3(0, 1, 0);
    vec3 right = cross(up, N);
    up = cross(N, right);

    float sampleDelta = 0.025;
    float nSamples = 0.0;
    for(float phi=0.0; phi<2.0*PI; phi+=sampleDelta)
    {
        for(float theta=0.0; theta<0.5*PI; theta+=sampleDelta)
        {
            vec3 tangentSample = vec3(sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta));
            vec3 sampleVec = tangentSample.x*right + tangentSample.y*up + tangentSample.z*N;
            irradiance += texture(u_EnvironmentMap, sampleVec).rgb*cos(theta)*sin(theta);
            nSamples++;
        }
    }

    irradiance *= PI/nSamples;
    f_Color = vec4(irradiance, 1.0);
}



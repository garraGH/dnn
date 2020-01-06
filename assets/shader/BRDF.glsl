#type vertex
#version 460 core
#line 1

in vec3 a_Position;
in vec2 a_TexCoord;

out vec2 v_TexCoord;

void main()
{
    v_TexCoord = a_TexCoord;
    gl_Position = vec4(a_Position, 1.0);
}

#type fragment
#version 460 core
#line 1

in vec2 v_TexCoord;
out vec4 f_Color;

const float PI = 3.14159265359;

float RadicalInverse_VDC(uint bits)
{
    bits = (bits<<16u) | (bits>>16u);
    bits = ((bits&0x55555555u)<<1u) | ((bits&0xAAAAAAAAu)>>1u);
    bits = ((bits&0x33333333u)<<2u) | ((bits&0xCCCCCCCCu)>>2u);
    bits = ((bits&0x0F0F0F0Fu)<<4u) | ((bits&0xF0F0F0F0u)>>4u);
    bits = ((bits&0x00FF00FFu)<<8u) | ((bits&0xFF00FF00u)>>8u);
    return float(bits)*2.3283064365386963e-10;
}

vec2 Hammersley(uint i, uint N)
{
    return vec2(float(i)/float(N), RadicalInverse_VDC(i));
}

vec3 ImportanceSampleGGX(vec2 Xi, vec3 N, float roughness)
{
    float a = roughness*roughness;
    float phi = 2.0*PI*Xi.x;
    float cosTheta = sqrt((1.0-Xi.y)/(1.0+(a*a-1.0)*Xi.y));
    float sinTheta = sqrt(1.0-cosTheta*cosTheta);

    vec3 H = vec3(cos(phi)*sinTheta, sin(phi)*sinTheta, cosTheta);
    vec3 up = abs(N.z)<0.999? vec3(0, 0, 1) : vec3(1, 0, 0);
    vec3 tangent = normalize(cross(up, N));
    vec3 bitangent = cross(N, tangent);

    vec3 sampleVec = tangent*H.x+bitangent*H.y+N*H.z;
    return normalize(sampleVec);
}

float GeometrySchlickGGX(float NdotV, float roughness)
{
    float a = roughness;
    float k = (a*a)/2.0;
    float nom = NdotV;
    float denom = NdotV*(1-k)+k;
    return nom/denom;
}

float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness)
{
    float NdotL = max(dot(N, L), 0.0);
    float NdotV = max(dot(N, V), 0.0);
    float ggxL = GeometrySchlickGGX(NdotL, roughness);
    float ggxV = GeometrySchlickGGX(NdotV, roughness);
    return ggxL*ggxV;
}

vec2 IntegrateBRDF(float NdotV, float roughness)
{
    float A = 0;
    float B = 0;
    vec3 V = vec3(sqrt(1.0-NdotV*NdotV), 0.0, NdotV);
    vec3 N = vec3(0, 0, 1);
    const uint SAMPLE_COUNT = 1024u;
    for(unsigned int i=0; i<SAMPLE_COUNT; i++)
    {
        vec2 Xi = Hammersley(i, SAMPLE_COUNT);
        vec3 H = ImportanceSampleGGX(Xi, N, roughness);
        vec3 L = normalize(2.0*dot(V, H)*H-V);

        float NdotL = max(L.z, 0.0);
        float NdotH = max(H.z, 0.0);
        float VdotH = max(dot(V, H), 0.0);

        if(NdotL>0.0)
        {
            float G = GeometrySmith(N, V, L, roughness);
            float G_Vis = (G*VdotH)/(NdotH*NdotV);
            float Fc = pow(1.0-VdotH, 5.0);
            A += (1.0-Fc)*G_Vis;
            B += Fc*G_Vis;
        }
    }

    A /= float(SAMPLE_COUNT);
    B /= float(SAMPLE_COUNT);

    return vec2(A, B);
}

void main()
{
    f_Color = vec4(IntegrateBRDF(v_TexCoord.x, v_TexCoord.y), 0, 1);
}

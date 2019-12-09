
#type vertex
#version 460 core
#line 1

// PerVertex
in vec3 a_Position;
in vec3 a_Normal;
in vec2 a_TexCoord;
// PerInstance
in mat4 a_MS2WS;
in vec3 a_Albedo;
in float a_Metallic;
in float a_Roughness;
in float a_Ao;


out VS_OUT
{
    vec3 FragPos;
    vec3 Normal;
    vec3 Albedo;
    vec2 TexCoord;
    float Metallic;
    float Roughness;
    float Ao;
}
v_Out;


layout(std140) uniform Transform
{
    mat4 WS2CS; // ViewProjection
}
u_Transform;

void main()
{
    vec4 pos = a_MS2WS*vec4(a_Position, 1);
    v_Out.TexCoord = a_TexCoord;
    v_Out.FragPos = pos.xyz;
    v_Out.Normal = mat3(a_MS2WS)*a_Normal;
    v_Out.Albedo = a_Albedo;
    v_Out.Metallic = a_Metallic;
    v_Out.Roughness = a_Roughness;
    v_Out.Ao = a_Ao;

    gl_Position = u_Transform.WS2CS*pos;
}

#type fragment
#version 460 core
#line 1

in VS_OUT
{
    vec3 FragPos;
    vec3 Normal;
    vec3 Albedo;
    vec2 TexCoord;
    float Metallic;
    float Roughness;
    float Ao;
}
f_In;

struct PointLight
{
    vec3 Position;
    float Padding0;
    vec3 Color;
    float Padding1;
};

layout(std140) uniform Light
{
    PointLight pLights[NUM_OF_POINTLIGHTS];
}
u_Light;

struct Camera
{
    vec3 Position;
};

uniform Camera u_Camera;

out vec4 f_Color;

#define PI 3.14159265

float DistributionGGX(vec3 N, vec3 H, float roughness)
{
    float r2 = roughness*roughness;
    float r4 = r2*r2;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH*NdotH;

    float nom = r4;
    float denom = (NdotH2*(r4-1.0)+1.0);
    denom *= PI*denom;

    return nom/max(denom, 0.0001);
}

float GeometrySchlickGGX(float NdotV, float roughness)
{
    float r = (roughness+1.0);
    float k = (r*r)/8.0;

    float nom = NdotV;
    float denom = NdotV*(1.0-k)+k;
    
    return nom/denom;
}

float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = GeometrySchlickGGX(NdotV, roughness);
    float ggx1 = GeometrySchlickGGX(NdotL, roughness);

    return ggx1*ggx2;
}

vec3 fresnelSchlick(float cosTheta, vec3 F0)
{
    return F0+(1.0-F0)*pow(1.0-cosTheta, 5.0);
}

void _ToneMap(inout vec3 color)
{
    color /= color+vec3(1.0f);
}

void _GammaCorrect(inout vec3 color)
{
    color = pow(color, vec3(1.0/2.2));
}

void main()
{
    vec3 N = normalize(f_In.Normal);
    vec3 V = normalize(u_Camera.Position-f_In.FragPos);
    
    vec3 F0 = vec3(0.04);
    F0 = mix(F0, f_In.Albedo, f_In.Metallic);

    vec3 Lo = vec3(0.0);
    for(int i=0; i<NUM_OF_POINTLIGHTS; i++)
    {
        vec3 lightDir = u_Light.pLights[i].Position-f_In.FragPos;
        vec3 L = normalize(lightDir);
        vec3 H = normalize(V+L);
        float distance = length(lightDir);
        float attenuation = 1.0;//(distance*distance);
        vec3 radiance = u_Light.pLights[i].Color*attenuation;

        float NDF = DistributionGGX(N, H, f_In.Roughness);
        float G = GeometrySmith(N, V, L, f_In.Roughness);
        vec3 F = fresnelSchlick(clamp(dot(H, V), 0.0, 1.0), F0);

        vec3 nominator = NDF*G*F;
        float denominator = 4*max(dot(N, V), 0.0)*max(dot(N, L), 0.0);
        vec3 specular = nominator/max(denominator, 0.0001);

        vec3 kS = F;
        vec3 kD = vec3(1.0)-kS;
        kD *= 1.0-f_In.Metallic;

        float NdotL = max(dot(N, L), 0.0);
        Lo += (kD*f_In.Albedo/PI+specular)*radiance*NdotL;
    }

    vec3 ambient = vec3(0.03)*f_In.Albedo*f_In.Ao;
    vec3 color = ambient+Lo;
    _ToneMap(color);
    _GammaCorrect(color);

    f_Color = vec4(color, 1.0);
//     f_Color = vec4(1, 0, 1, 1);
}


#type vertex
#version 460 core
#line 1

// PerVertex
in vec3 a_Position;
in vec3 a_Normal;
in vec2 a_TexCoord;
// PerInstance
in mat4 a_MS2WS;
#ifndef ALBEDO_MAP
in vec3 a_Albedo;
#endif

#ifndef METALLIC_MAP
in float a_Metallic;
#endif

#ifndef ROUGHNESS_MAP
in float a_Roughness;
#endif

#ifndef AO_MAP
in float a_Ao;
#endif


out VS_OUT
{
    vec3 FragPos;
    vec3 Normal;
    vec2 TexCoord;

#ifndef ALBEDO_MAP
    vec3 Albedo;
#endif

#ifndef METALLIC_MAP
    float Metallic;
#endif

#ifndef ROUGHNESS_MAP
    float Roughness;
#endif

#ifndef AO_MAP
    float Ao;
#endif
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
#ifndef ALBEDO_MAP
    v_Out.Albedo = a_Albedo;
#endif

#ifndef METALLIC_MAP
    v_Out.Metallic = a_Metallic;
#endif

#ifndef ROUGHNESS_MAP
    v_Out.Roughness = a_Roughness;
#endif

#ifndef AO_MAP
    v_Out.Ao = a_Ao;
#endif

    gl_Position = u_Transform.WS2CS*pos;
}

#type fragment
#version 460 core
#line 1

in VS_OUT
{
    vec3 FragPos;
    vec3 Normal;
    vec2 TexCoord;

#ifndef ALBEDO_MAP
    vec3 Albedo;
#endif

#ifndef METALLIC_MAP
    float Metallic;
#endif

#ifndef ROUGHNESS_MAP
    float Roughness;
#endif

#ifndef AO_MAP
    float Ao;
#endif
}
f_In;

#ifdef NORMAL_MAP
uniform sampler2D u_NormalMap;
#endif

#ifdef ALBEDO_MAP
uniform sampler2D u_AlbedoMap;
#endif

#ifdef METALLIC_MAP
uniform sampler2D u_MetallicMap;
#endif

#ifdef ROUGHNESS_MAP
uniform sampler2D u_RoughnessMap;
#endif

#ifdef AO_MAP
uniform sampler2D u_AoMap;
#endif

#ifdef IRRADIANCE_MAP
uniform sampler2D u_IrradianceMap;
#endif

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


layout(location = 0) out vec4 f_Color;
layout(location = 1) out vec4 f_BrightColor;

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

vec3 _Normal()
{
    vec3 N = normalize(f_In.Normal);
    #ifdef NORMAL_MAP
    {
        vec3 tangentNormal = texture(u_NormalMap, f_In.TexCoord).xyz*2.0-1.0;
        vec3 Q1 = dFdx(f_In.FragPos);
        vec3 Q2 = dFdy(f_In.FragPos);
        vec2 st1 = dFdx(f_In.TexCoord);
        vec2 st2 = dFdy(f_In.TexCoord);

        vec3 T = normalize(Q1*st2.t-Q2*st1.t);
        vec3 B = -normalize(cross(N, T));
        mat3 TBN = mat3(T, B, N);

        return normalize(TBN*tangentNormal);
    }
    #else
    {
        return N;
    }
    #endif
}

float _Roughness()
{
    #ifdef ROUGHNESS_MAP
    {
        return texture(u_RoughnessMap, f_In.TexCoord).r;
    }
    #else
    {
        return f_In.Roughness;
    }
    #endif
}

float _Metallic()
{
    #ifdef METALLIC_MAP
    {
        return texture(u_MetallicMap, f_In.TexCoord).r;
    }
    #else
    {
        return f_In.Metallic ;
    }
    #endif
}

float _Ao()
{ 
    #ifdef AO_MAP
    {
        return texture(u_MetallicMap, f_In.TexCoord).r;
    }
    #else
    {
        return f_In.Metallic ;
    }
    #endif 
}

vec3 _Albedo()
{
    #ifdef ALBEDO_MAP
    {
        return pow(texture(u_AlbedoMap, f_In.TexCoord).rgb, vec3(2.2));
    }
    #else
    {
        return pow(f_In.Albedo, vec3(2.2));
    }
    #endif
}

vec2 _UVofNormal(vec3 N)
{
    return vec2(1);
}

vec3 _Irradiance(vec3 N)
{
    #ifdef IRRADIANCE_MAP
    {
        return texture(u_IrradianceMap, _UVofNormal(N)).rgb;
    }
    #else
    {
        return vec3(1.0);
    }
    #endif
}
        
vec3 _ViewDirection()
{
    return normalize(u_Camera.Position-f_In.FragPos);
}

void main()
{
    vec3 N = _Normal();
    vec3 V = _ViewDirection();
    
    vec3 F0 = vec3(0.04);
    F0 = mix(F0, _Albedo(), _Metallic());

    vec3 Lo = vec3(0.0);
    for(int i=0; i<NUM_OF_POINTLIGHTS; i++)
    {
        vec3 lightDir = u_Light.pLights[i].Position-f_In.FragPos;
        vec3 L = normalize(lightDir);
        vec3 H = normalize(V+L);
        float distance = length(lightDir);
        float attenuation = 1.0;//(distance*distance);
        vec3 radiance = u_Light.pLights[i].Color*attenuation;

        float NDF = DistributionGGX(N, H, _Roughness());
        float G = GeometrySmith(N, V, L, _Roughness());  
        vec3 F = fresnelSchlick(clamp(dot(H, V), 0.0, 1.0), F0);

        vec3 nominator = NDF*G*F;
        float denominator = 4*max(dot(N, V), 0.0)*max(dot(N, L), 0.0);
        vec3 specular = nominator/max(denominator, 0.0001);

        vec3 kS = F;
        vec3 kD = (1.0-kS)*(1.0-_Metallic());

        float NdotL = max(dot(N, L), 0.0);
        Lo += (kD*_Albedo()/PI+specular)*radiance*NdotL;
    }

    vec3 kS = fresnelSchlick(max(dot(N, V), 0.0), F0);
    vec3 kD = (1.0-kS)*(1.0-_Metallic());
    vec3 diffuse = _Irradiance(N)*_Albedo();
    vec3 ambient = kD*diffuse*_Ao();
    vec3 color = ambient+Lo;
    _ToneMap(color);
    _GammaCorrect(color);
    f_Color = vec4(color, 1.0);
}

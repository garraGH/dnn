
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

#ifdef IRRADIANCE_DIFFUSE_MAP
uniform samplerCube u_IrradianceMap;
#endif

#ifdef IRRADIANCE_SPECULAR_MAP
uniform sampler2D u_LUTofBRDF;
uniform samplerCube u_PrefilterMap;
#endif

struct PointLight
{
    vec3 Position;
    float Padding0;
    vec3 Color;
    float Padding1;
};

#if NUM_OF_POINTLIGHTS>0
layout(std140) uniform Lights
{
    PointLight pLights[NUM_OF_POINTLIGHTS];
}
u_Lights;
#endif

struct Camera
{
    vec3 Position;
};

uniform Camera u_Camera;


out vec4 f_Color;

#define PI 3.14159265

float DistributionGGX(vec3 N, vec3 H, float a)
{
    float a2 = a*a;

    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH*NdotH;

    float nom = a2;
    float c = NdotH2*(a2-1.0)+1.0;
    float denom = PI*c*c;

    return nom/max(denom, 0.0001);
}

float GeometrySchlickGGX(float NdotV, float k)
{
    float nom = NdotV;
    float denom = NdotV*(1.0-k)+k;
    
    return nom/denom;
}

float GeometrySmith(vec3 N, vec3 V, vec3 L, float k)
{
    float NdotL = max(dot(N, L), 0.0);
    float NdotV = max(dot(N, V), 0.0);
    float ggx1 = GeometrySchlickGGX(NdotL, k);
    float ggx2 = GeometrySchlickGGX(NdotV, k);

    return ggx1*ggx2;
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
        return texture(u_AoMap, f_In.TexCoord).r;
    }
    #else
    {
        return f_In.Ao ;
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


vec3 _Irradiance(vec3 N)
{
    #ifdef IRRADIANCE_DIFFUSE_MAP
    {
        return texture(u_IrradianceMap, N).rgb;
    }
    #else
    {
        return vec3(0.0);
    }
    #endif
}
        
vec3 _ViewDirection()
{
    return normalize(u_Camera.Position-f_In.FragPos);
}

vec3 fresnelSchlick(float cosTheta, vec3 F0)
{
    return F0+(1.0-F0)*pow(1.0-cosTheta, 5.0);
}

vec3 fresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness)
{
    return F0+(max(vec3(1.0-roughness), F0)-F0)*pow(1-cosTheta, 5.0);
}

vec3 _PrefilteredColor(vec3 R)
{
#ifdef IRRADIANCE_SPECULAR_MAP
    const float MAX_REFLECTION_LOD = 4.0;
    return textureLod(u_PrefilterMap, R, _Roughness()*MAX_REFLECTION_LOD).rgb;
#else
    return vec3(0);
#endif
}

vec3 _Brdf(vec2 uv, vec3 F)
{
#ifdef IRRADIANCE_SPECULAR_MAP
    vec2 brdf = texture(u_LUTofBRDF, uv).rg;
    return F*brdf.x+brdf.y;

#else
    return vec3(0);
#endif
}

void main()
{
    vec3 N = _Normal();
    vec3 V = _ViewDirection();
    vec3 R = reflect(-V, N);

    vec3 F0 = vec3(0.04);
    F0 = mix(F0, _Albedo(), _Metallic());

    // direct lighting
    float a = _Roughness();
    float a2 = a*a;
    vec3 Lo = vec3(0.0);
#if NUM_OF_POINTLIGHTS>0
    for(int i=0; i<NUM_OF_POINTLIGHTS; i++)
    {
        vec3 lightDir = u_Lights.pLights[i].Position-f_In.FragPos;
        vec3 L = normalize(lightDir);
        vec3 H = normalize(V+L);
        float distance = length(lightDir);
        float attenuation = 1.0/(distance*distance);
        vec3 radiance = u_Lights.pLights[i].Color*attenuation;

        float r = a2+1;
        float k = (r*r)/8;
        float NdotV = max(dot(N, V), 0.0);
        float NdotL = max(dot(N, L), 0.0);
        float HdotV = max(dot(H, V), 0.0);
        float D = DistributionGGX(N, H, a2);
        float G = GeometrySmith(N, V, L, k);
        vec3  F = fresnelSchlick(HdotV, F0);

        vec3 nom = D*G*F;
        float denom = 4*NdotV*NdotL+0.001;
        vec3 specular = nom/denom;

        vec3 ks = F;
        vec3 kd = (1-ks)*(1-_Metallic());

        vec3 diffuse = kd*_Albedo()/PI;
        Lo += (diffuse+specular)*radiance*NdotL;
    }
#endif

    // indirect lighting
    vec3 F = fresnelSchlickRoughness(max(dot(N, V), 0.0), F0, _Roughness());
    vec3 ks = F;
    vec3 kd = (1-ks)*(1-_Metallic());

    vec3 diffuse = _Irradiance(N)*_Albedo();
    vec2 uv = vec2(max(dot(N, V), 0.0), _Roughness());
    vec3 specular = _PrefilteredColor(R)*_Brdf(uv, F);
    vec3 ambient = (kd*diffuse+specular)*_Ao();

    vec3 color = ambient+Lo;

    _ToneMap(color);
    _GammaCorrect(color);

    f_Color = vec4(color, 1.0);
}

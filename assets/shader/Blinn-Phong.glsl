#type vertex
#version 460 core
#line 1

// MS: ModelSpace
// WS: WorldSpace
// VS: ViewSpace
// CS: ClipSpace
// NDC: NormalizeDeviceCoordinate
// TS: TagentSpace

in vec3 a_PositionMS;
in vec3 a_NormalMS;
in vec3 a_TangentMS;
in vec2 a_TexCoord;
#ifdef INSTANCE
in mat4 a_MS2WS;
#endif

struct Camera
{
    vec3 PositionWS;
};

// uniform 
uniform mat4 u_MS2WS;
uniform Camera u_Camera;

// uniform buffer
layout(std140) uniform Transform
{
    mat4 WS2CS;
}
u_Transform;


out VS_OUT
{
    vec3 FragPosWS;
    vec2 TexCoord;
    vec3 CameraPosWS;
#ifndef NORMAL_MAP
    vec3 NormalWS;
#else
    mat3 TS2WS;
#endif
#ifdef DISPLACEMENT_MAP
    vec3 CameraPosTS;
    vec3 FragPosTS;
#endif
}
v_Out;

void main()
{
    v_Out.TexCoord = a_TexCoord;
    v_Out.CameraPosWS = u_Camera.PositionWS;
    mat4 ms2ws = u_MS2WS;
    #ifdef INSTANCE
    {
        ms2ws = a_MS2WS;
    }
    #endif
    vec4 pos = ms2ws*vec4(a_PositionMS, 1);
    v_Out.FragPosWS = pos.xyz;
    mat3 _ms2ws = mat3(ms2ws);
    #ifdef NORMAL_MAP
    {
        vec3 t = normalize(_ms2ws*a_TangentMS);
        vec3 n = normalize(_ms2ws*a_NormalMS);
        vec3 b = cross(n, t);
        v_Out.TS2WS = mat3(t, b, n);
    }
    #else
    {
        v_Out.NormalWS = transpose(inverse(_ms2ws))*a_NormalMS;
    }
    #endif
    #ifdef DISPLACEMENT_MAP
    {
        mat3 ws2ts;
        #ifdef NORMAL_MAP
        {
            ws2ts = transpose(v_Out.TS2WS);
        }
        #else
        {
            vec3 t = normalize(_ms2ws*a_TangentMS);
            vec3 n = normalize(_ms2ws*a_NormalMS);
            vec3 b = cross(n, t);
            ws2ts = transpose(mat3(t, b, n));
        }
        #endif
        v_Out.CameraPosTS = ws2ts*u_Camera.PositionWS;
        v_Out.FragPosTS = ws2ts*v_Out.FragPosWS;
    }
    #endif

    gl_Position = u_Transform.WS2CS*pos;
}

//---------------------------------------------------------------------------------
#type fragment
#version 460 core
#line 1

struct Material
{
#ifdef DIFFUSE_REFLECTANCE
    vec3 DiffuseReflectance;
#endif

#ifdef DIFFUSE_MAP
    sampler2D DiffuseMap;
#endif

#ifdef SPECULAR_REFLECTANCE
    vec3 SpecularReflectance;
#endif

#ifdef SPECULAR_MAP
    sampler2D SpecularMap;
#endif 

#if defined SPECULAR_REFLECTANCE || defined SPECULAR_MAP
    float Shininess;
#endif

#if defined EMISSIVE_COLOR || defined EMISSIVE_MAP
    float EmissiveIntensity;
#endif

#ifdef EMISSIVE_COLOR
    vec3 EmissiveColor;
#endif

#ifdef EMISSIVE_MAP
    sampler2D EmissiveMap;
#endif

#ifdef NORMAL_MAP
    sampler2D NormalMap;
#endif

#ifdef HEIGHT_MAP
    sampler2D HeightMap;
#endif

#ifdef DISPLACEMENT_MAP
    sampler2D DisplacementMap;
    float DisplacementScale;
#endif

};

struct DirectionalLight
{
    vec3 Color;
    float Padding0;
    vec3 DirectionWS;
    float Intensity;
};

struct PointLight
{
    vec3 Color;
    float Padding0;
    vec3 PositionWS;
    float Padding1;
    vec3 AttenuationCoefficients; // constant, linear, quadratic
    float Intensity;
};

struct SpotLight
{
    vec3 Color;
    float CosInnerCone;
    vec3 PositionWS;
    float CosOuterCone;
    vec3 DirectionWS;
    float Padding0;
    vec3 AttenuationCoefficients; // constant, linear, quadratic
    float Intensity;
};



// uniform buffer
layout(std140) uniform Light
{
    DirectionalLight dLight;    // 2*16
    PointLight pLight;          // 3*16
    SpotLight sLight;           // 4*16
    SpotLight fLight;           // 4*16
}
u_Light;


uniform vec3 u_AmbientColor = vec3(0.2f);
uniform float u_BloomThreshold = 1.0f;
uniform Material u_Material;

in VS_OUT
{
    vec3 FragPosWS;
    vec2 TexCoord;
    vec3 CameraPosWS;
#ifndef NORMAL_MAP
    vec3 NormalWS;
#else
    mat3 TS2WS;
#endif

#ifdef DISPLACEMENT_MAP
    vec3 CameraPosTS;
    vec3 FragPosTS;
#endif
}
f_In;


layout(location = 0) out vec4 f_Color;
layout(location = 1) out vec4 f_ColorBright;


vec2 _TexCoord()
{
    #ifndef DISPLACEMENT_MAP
    {
        return f_In.TexCoord;
    }
    #else
    {
//         float depth = texture(u_Material.DisplacementMap, f_In.TexCoord).r;
//         vec3 viewDirTS = normalize(f_In.CameraPosTS-f_In.FragPosTS);
//         vec2 P = viewDirTS.xy/viewDirTS.z*(depth*u_Material.DisplacementScale);
//         return f_In.TexCoord-P;
        const float minLayers = 8.0;
        const float maxLayers = 32.0;
        vec3 normalTS = vec3(0, 0, 1);
        vec3 viewDirTS = normalize(f_In.CameraPosTS-f_In.FragPosTS);
        float numLayers = mix(maxLayers, minLayers, abs(dot(normalTS, viewDirTS)));
        float layerDepth = 1.0/numLayers;
        float currentLayerDepth = 0.0;
        vec2 P = viewDirTS.xy/viewDirTS.z*u_Material.DisplacementScale;
        vec2 deltaTexCoords = P/numLayers;

        vec2 currentTexCoords = f_In.TexCoord;
        float currentDisplacementMapValue = texture(u_Material.DisplacementMap, currentTexCoords).r;
        while(currentLayerDepth<currentDisplacementMapValue)
        {
            currentTexCoords -= deltaTexCoords;
            currentDisplacementMapValue = texture(u_Material.DisplacementMap, currentTexCoords).r;
            currentLayerDepth += layerDepth;
        }
        
        vec2 prevTexCoords = currentTexCoords+deltaTexCoords;
        float afterDepth = currentDisplacementMapValue-currentLayerDepth;
        float beforeDepth = texture(u_Material.DisplacementMap, prevTexCoords).r-currentLayerDepth+layerDepth;
        float weight = afterDepth/(afterDepth-beforeDepth);
        vec2 finalTexCoords = prevTexCoords*weight+currentTexCoords*(1-weight);
        if(finalTexCoords.x<0.0||finalTexCoords.x>1.0||finalTexCoords.y<0.0||finalTexCoords.y>1.0)
        {
            discard;
        }
        return finalTexCoords;
    }
    #endif
    
}

vec3 _Normal()
{
    #ifndef NORMAL_MAP
    {
        return f_In.NormalWS;
    }
    #else
    {
        vec3 normalTS =  texture(u_Material.NormalMap, _TexCoord()).xyz;
        normalTS = normalize(normalTS*2.0-1.0);
        vec3 normalWS = normalize(f_In.TS2WS*normalTS);
        return normalWS;
    }
    #endif
}

vec3 _DiffuseReflectance()
{
    vec3 diffuseReflectance = vec3(1.0f);
    #ifdef DIFFUSE_REFLECTANCE
    {
        diffuseReflectance = u_Material.DiffuseReflectance;
    }
    #endif
    #ifdef DIFFUSE_MAP
    {
        diffuseReflectance *= texture(u_Material.DiffuseMap, _TexCoord()).rgb;
    }
    #endif
    return diffuseReflectance;
}

vec3 _SpecularReflectance()
{
    vec3 specularReflectance = vec3(1.0f);
    #ifdef SPECULAR_REFLECTANCE
    {
        specularReflectance = u_Material.SpecularReflectance;
    }
    #endif 
    #ifdef SPECULAR_MAP
    {
        specularReflectance *= texture(u_Material.SpecularMap, _TexCoord()).rgb;
    }
    #endif 
    return specularReflectance;
}

vec3 _DiffuseReflectance(in vec3 lightDir)
{
    #if !defined DIFFUSE_REFLECTANCE && !defined DIFFUSE_MAP
    {
        return vec3(0.0f);
    }
    #else
    {
        float strength = max(dot(_Normal(), lightDir), 0.0f);
        return strength*_DiffuseReflectance();
    }
    #endif
}

vec3 _SpecularReflectance(in vec3 lightDir)
{
    #if !defined SPECULAR_REFLECTANCE && !defined SPECULAR_MAP
    {
        return vec3(0.0f);
    }
    #else
    {
        vec3 normal = _Normal();
        vec3 viewDir = normalize(f_In.CameraPosWS-f_In.FragPosWS);
        float strengthBase = 0.0f;
        #ifdef PHONG
        {
            vec3 reflectDir = -reflect(lightDir, normal);
            strengthBase = max(dot(reflectDir, viewDir), 0.0f);
        }
        #else // default: Blinn-Phong
        {
            vec3 bisector = normalize(lightDir+viewDir);
            strengthBase = max(dot(bisector, normal), 0.0f);
        }
        #endif
        float strength = pow(strengthBase, u_Material.Shininess);
        return strength*_SpecularReflectance();
    }
    #endif
}

vec3 _Reflectance(in vec3 lightDir)
{
    return _DiffuseReflectance(lightDir)+_SpecularReflectance(lightDir);
}

vec3 Ambient()
{
    return u_AmbientColor*_DiffuseReflectance();
}

vec3 Emission()
{
    #if !defined EMISSIVE_COLOR && !defined EMISSIVE_MAP
    {
        return vec3(0.0f);
    }
    #else
    {
        vec3 emissiveColor = vec3(0.0f);
        #ifdef EMISSIVE_COLOR
        {
            emissiveColor = u_Material.EmissiveColor;
        }
        #endif
        #ifdef EMISSIVE_MAP
        {
            emissiveColor += texture(u_Material.EmissiveMap, _TexCoord()).rgb;
        }
        #endif
        
        return u_Material.EmissiveIntensity*emissiveColor;
    }
    #endif
}

float _Attenuation(in vec3 lightPosWS, in vec3 coef)
{
    float distance = length(lightPosWS-f_In.FragPosWS);
    return 1.0/(coef.x+coef.y*distance+coef.z*(distance*distance));
}

float _SpotIntensity(in vec3 spotDir, in vec3 lightDir, float inner, float outer)
{
    float theta = dot(spotDir, lightDir);
    return clamp((theta-outer)/(inner-outer), 0.0, 1.0);
}

vec3 ColorFrom(in DirectionalLight light)
{
    vec3 lightDir = -normalize(light.DirectionWS);
    vec3 reflectance = _Reflectance(lightDir);
    return reflectance*light.Color*light.Intensity;
}

vec3 ColorFrom(in PointLight light)
{
    vec3 lightDir = normalize(light.PositionWS-f_In.FragPosWS);
    vec3 reflectance = _Reflectance(lightDir);
    float attenuation = _Attenuation(light.PositionWS, light.AttenuationCoefficients);
    return attenuation*reflectance*light.Color*light.Intensity;
}

vec3 ColorFrom(in SpotLight light)
{
    vec3 lightDir = normalize(light.PositionWS-f_In.FragPosWS);
    vec3 reflectance = _Reflectance(lightDir);
    float attenuation = _Attenuation(light.PositionWS, light.AttenuationCoefficients);
    float intensity = _SpotIntensity(-normalize(light.DirectionWS), lightDir, light.CosInnerCone, light.CosOuterCone);
    return attenuation*intensity*reflectance*light.Color*light.Intensity;
}
                                       

void main()
{
    vec3 color = Ambient();
    color += Emission();
    color += ColorFrom(u_Light.dLight);
    color += ColorFrom(u_Light.pLight);
    color += ColorFrom(u_Light.sLight);
    color += ColorFrom(u_Light.fLight);
    f_Color= vec4(color, 1.0);

    float brightness = dot(color, vec3(0.2126, 0.7152, 0.0722));
    f_ColorBright = vec4(step(u_BloomThreshold, brightness)*color, 1.0);
}




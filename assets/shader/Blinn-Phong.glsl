#type vertex
#version 460 core

layout(location = 0) in vec3 a_Position;
layout(location = 1) in vec3 a_Normal;
layout(location = 2) in vec2 a_TexCoord;
#ifdef INSTANCE
    layout(location = 3) in mat4 a_Model2World;
#endif
uniform mat4 u_Model2World;

layout(std140) uniform Transform
{
    mat4 world2Clip;
}
u_Transform;


out VS_OUT
{
    vec3 normal;
    vec3 fragPos;
    vec2 texCoord;
}
v_out;

void main()
{
    v_out.texCoord = a_TexCoord;
    mat4 model2World = u_Model2World;
    #ifdef INSTANCE
    {
        model2World = a_Model2World;
    }
    #endif
    vec4 pos = model2World*vec4(a_Position, 1);
//     v_out.normal = mat3(transpose(inverse(model2World)))*a_Normal;
//     v_out.normal = (model2World*vec4(a_Normal, 0)).xyz;
    v_out.normal = mat3(model2World)*a_Normal;
    v_out.fragPos = pos.xyz;

    gl_Position = u_Transform.world2Clip*pos;
}

#type fragment
#version 460 core

layout(std140) uniform Camera1
{
    vec3 position;
}
u_Camera1;



struct Material
{
    vec3 diffuseReflectance;

#ifdef SPECLUAR_REFLECTANCE
    vec3 specularReflectance;
    float shininess;        
#endif

#ifdef EMISSIVE_COLOR
    vec3 emissiveColor;
#endif

#ifdef DIFFUSE_MAP
    sampler2D diffuseMap;
#endif

#ifdef SPECULAR_MAP
    sampler2D specularMap;
#endif 

#ifdef EMISSIVE_MAP
    sampler2D emissiveMap;
#endif

#ifdef NORMAL_MAP
    sampler2D normalMap;
#endif

#ifdef HEIGHT_MAP
    sampler2D heightMap;
#endif

};

struct DirectionalLight
{
    vec3 color;
    vec3 direction;
};

struct PointLight
{
    vec3 color;
    vec3 position;
    vec3 attenuationCoefficients; // constant, linear, quadratic
};

struct SpotLight
{
    vec3 color;
    vec3 position;
    vec3 direction;
    vec3 attenuationCoefficients; // constant, linear, quadratic
    float cosInnerCone;
    float cosOuterCone;
};


struct Camera
{
    vec3 position;
};


layout(std140) uniform Light
{
    DirectionalLight dLight;
    PointLight pLight;
    SpotLight sLight;
    SpotLight fLight;
}
u_Light;


uniform Material u_Material;
uniform Camera u_Camera;
uniform vec3 u_AmbientColor = vec3(0.2f);

in VS_OUT
{
    vec3 normal;
    vec3 fragPos;
    vec2 texCoord;
}
f_in;

out vec4 f_Color;


vec3 _DiffuseReflectance()
{
    vec3 diffuseReflectance = u_Material.diffuseReflectance;
    #ifdef DIFFUSE_MAP
    {
        diffuseReflectance *= texture(u_Material.diffuseMap, f_in.texCoord).rgb;
    }
    #endif
    return diffuseReflectance;
}

vec3 _SpecularReflectance()
{
    #ifndef SPECULAR_REFLECTANCE
    {
        return vec3(0.0f);
    }
    #else
    {
        vec3 specularReflectance = u_Material.specularReflectance;
        #ifdef SPECULAR_MAP
        {
            specularReflectance *= texture(u_Material.specularMap, f_in.texCoord).rgb;
        }
        #endif 
        return specularReflectance;
    }
    #endif
}

vec3 _Normal()
{
    return f_in.normal;
}

vec3 _DiffuseReflectance(in vec3 lightDir)
{
    float strength = max(dot(_Normal(), lightDir), 0.0f);
    return strength*_DiffuseReflectance();
}

vec3 _SpecularReflectance(in vec3 lightDir)
{
    #ifndef SPECLUAR_REFLECTANCE
    {
        return vec3(0.0f);
    }
    #else
    {
        vec3 normal = _Normal();
        vec3 viewDir = normalize(u_Camera.position-f_in.fragPos);
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
        float strength = pow(strengthBase, u_Material.shininess);
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
    #ifndef EMISSIVE_COLOR
    {
        return vec3(0.0f);
    }
    #else
    {
        vec3 emissiveColor = u_Material.emissiveColor;
        #ifdef EMISSIVE_MAP
        {
            emissiveColor *= texture(u_Material.emissiveMap, f_in.texCoord).rgb;
        }
        #endif
        return emissiveColor;
    }
    #endif
}

float _Attenuation(in vec3 pos, in vec3 coef)
{
    float distance = length(pos-f_in.fragPos);
    return 1.0/(coef.x+coef.y*distance+coef.z*(distance*distance));
}

float _SpotIntensity(in vec3 center, in vec3 dir, float inner, float outer)
{
    float theta = dot(dir, center);
    return step(inner, theta);
    return clamp(1-(theta-outer)/(inner-outer), 0.0, 1.0);
}

vec3 ColorFrom(in DirectionalLight light)
{
    vec3 lightDir = -normalize(light.direction);
    vec3 reflectance = _Reflectance(lightDir);
    return reflectance*light.color;
}

vec3 ColorFrom(in PointLight light)
{
    vec3 lightDir = normalize(light.position-f_in.fragPos);
    vec3 reflectance = _Reflectance(lightDir);
    float attenuation = _Attenuation(light.position, light.attenuationCoefficients);
    return attenuation*reflectance*light.color;
}

vec3 ColorFrom(in SpotLight light)
{
    vec3 lightDir = normalize(light.position-f_in.fragPos);
    vec3 reflectance = _Reflectance(lightDir);
    float attenuation = _Attenuation(light.position, light.attenuationCoefficients);
    float intensity = _SpotIntensity(-normalize(light.direction), lightDir, light.cosInnerCone, light.cosOuterCone);
    return attenuation*intensity*reflectance*light.color;
}
                                       

void main()
{

    vec3 color = Ambient();
    color += Emission();
    color += ColorFrom(u_Light.dLight);
    color += ColorFrom(u_Light.pLight);
    color += ColorFrom(u_Light.sLight);
    color += ColorFrom(u_Light.fLight);

    f_Color = vec4(color, 1.0);
}




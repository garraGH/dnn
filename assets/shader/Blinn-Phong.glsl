#type vertex
#version 460 core
layout(location = 0) in vec3 a_Position;
layout(location = 1) in vec3 a_Normal;
layout(location = 2) in vec2 a_TexCoord;
layout(location = 3) in mat4 a_Model2World;
uniform mat4 u_World2Clip;
uniform mat4 u_Model2World;


out vec3 v_Normal;
out vec3 v_FragPos;
out vec2 v_TexCoord;

void main()
{
    v_Normal = a_Normal;
    v_TexCoord = a_TexCoord;
    v_FragPos = vec3(u_Model2World*a_Model2World*vec4(a_Position, 1.0));
    gl_Position = u_World2Clip*vec4(v_FragPos, 1.0);
}

#type fragment
#version 460 core

struct Material
{
    vec3 ambientReflectance;
    vec3 diffuseReflectance;
    vec3 specularReflectance;
    vec3 emissiveColor;
    float shininess;
    sampler2D diffuseMap;
    sampler2D specularMap;
    sampler2D emissiveMap;
};

struct Light
{
    vec3 color;
    vec3 position;
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
    vec3 attenuationCoefficients; // constant, linear, quadratic
    vec3 direction;
    float innerCone;
    float outerCone;
};


struct Camera
{
    vec3 position;
};



uniform Material u_Material;
uniform Light u_Light;
uniform Camera u_Camera;

#define MAX_DIRECTIONAL_LIGHTS 4
#define MAX_POINT_LIGHTS 100
#define MAX_SPOT_LIGHTS 100

uniform ivec3 u_NumOfLights = {1, 1, 1}; // (directional, point, spot)
uniform DirectionalLight u_DirectionalLight[MAX_DIRECTIONAL_LIGHTS];
uniform PointLight u_PointLight[MAX_POINT_LIGHTS];
uniform SpotLight u_SpotLight[MAX_SPOT_LIGHTS];
uniform SpotLight u_FlashLight;
uniform vec3 u_AmbientColor = vec3(0.2f);

in vec3 v_Normal;
in vec3 v_FragPos;
in vec2 v_TexCoord;
out vec4 f_Color;


vec3 CalculateAmbient()
{
    vec3 diffuseTexel = texture(u_Material.diffuseMap, v_TexCoord).rgb;
    return u_AmbientColor*u_Material.ambientReflectance*diffuseTexel;
}

vec3 CalculateEmission()
{
    return u_Material.emissiveColor*texture(u_Material.emissiveMap, v_TexCoord).rgb;
}

void _CalculateDiffuseAndSpecular(in vec3 lightDir, out vec3 diffuse, out vec3 specular)
{
    vec3 normal = normalize(v_Normal);
    vec3 viewDir = normalize(u_Camera.position-v_FragPos);
    vec3 diffuseTexel = texture(u_Material.diffuseMap, v_TexCoord).rgb;
    vec3 specularTexel = texture(u_Material.specularMap, v_TexCoord).rgb;
    vec3 bisector = normalize(lightDir+viewDir);
    float diff = max(dot(normal, lightDir), 0.0f);
    float spec = pow(max(dot(bisector, normal), 0.0f), u_Material.shininess);
    diffuse = diff*u_Material.diffuseReflectance*diffuseTexel;
    specular = spec*u_Material.specularReflectance*specularTexel;
}

vec3 CalculateDirectionalLight(in DirectionalLight light)
{
    vec3 diffuse;
    vec3 specular;
    vec3 lightDir = -normalize(light.direction);
    _CalculateDiffuseAndSpecular(lightDir, diffuse, specular);
    return (diffuse+specular)*light.color;
}

vec3 CalculatePointLight(in PointLight light)
{
    vec3 diffuse;
    vec3 specular;
    vec3 lightDir = normalize(light.position-v_FragPos);
    _CalculateDiffuseAndSpecular(lightDir, diffuse, specular);

    float distance = length(light.position-v_FragPos);
    float attenuation = 1.0/(light.attenuationCoefficients.x+light.attenuationCoefficients.y*distance+light.attenuationCoefficients.z*(distance*distance));
    return attenuation*(diffuse+specular)*light.color;
}

vec3 CalculateSpotLight(in SpotLight light)
{
    vec3 diffuse;
    vec3 specular;
    vec3 lightDir = normalize(light.position-v_FragPos);
    _CalculateDiffuseAndSpecular(lightDir, diffuse, specular);

    float distance = length(light.position-v_FragPos);
    float attenuation = 1.0/(light.attenuationCoefficients.x+light.attenuationCoefficients.y*distance+light.attenuationCoefficients.z*(distance*distance));

    float theta = dot(lightDir, normalize(-light.direction));
    float epsilon = light.innerCone-light.outerCone;
    float intensity = clamp((theta-light.outerCone)/epsilon, 0.0, 1.0);

    return attenuation*intensity*(diffuse+specular)*light.color;
}
                                       

void main()
{
    vec3 color = CalculateAmbient();
    for(int i=0; i<u_NumOfLights.x; i++)
    {
        color += CalculateDirectionalLight(u_DirectionalLight[i]);
    }
    for(int i=0; i<u_NumOfLights.y; i++)
    {
        color += CalculatePointLight(u_PointLight[i]);
    }
    for(int i=0; i<u_NumOfLights.z; i++)
    {
        color += CalculateSpotLight(u_SpotLight[i]);
    }
    color += CalculateSpotLight(u_FlashLight);
    color += CalculateEmission();

    f_Color = vec4(color, 1.0);
}




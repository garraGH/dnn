#type vertex
#version 460 core
layout(location = 0) in vec3 a_Position;
layout(location = 1) in vec3 a_Normal;
layout(location = 2) in vec2 a_TexCoord;
uniform mat4 u_World2Clip;
uniform mat4 u_Model2World;

out vec3 v_Normal;
out vec3 v_FragPos;
out vec2 v_TexCoord;

void main()
{
    v_Normal = a_Normal;
    v_TexCoord = a_TexCoord;
    v_FragPos = vec3(u_Model2World*vec4(a_Position, 1.0));
    gl_Position = u_World2Clip*vec4(v_FragPos, 1.0);
}

#type fragment
#version 460 core

struct Material
{
    vec3 ambientColor;
    vec3 diffuseColor;
    vec3 specularColor;
    float shininess;
    sampler2D diffuseMap;
    sampler2D specularMap;
    sampler2D emissionMap;
};

struct Light
{
    vec3 position;
    vec3 ambientColor;
    vec3 diffuseColor;
    vec3 specularColor;
};

struct Camera
{
    vec3 position;
};



uniform Material u_Material;
uniform Light u_Light;
uniform Camera u_Camera;

in vec3 v_Normal;
in vec3 v_FragPos;
in vec2 v_TexCoord;
out vec4 f_Color;

void main()
{
    // Ambient
    vec3 ambient = u_Light.ambientColor*u_Material.ambientColor;

    // Diffuse
    vec3 norm = normalize(v_Normal);
    vec3 lightDir = normalize(u_Light.position-v_FragPos);
    float diffuseStrength = max(dot(norm, lightDir), 0.0f);
    vec3 diffuseTexel = texture(u_Material.diffuseMap, v_TexCoord).rgb;
    vec3 diffuse = diffuseStrength*u_Light.diffuseColor*u_Material.diffuseColor*diffuseTexel;

    // Specular
    vec3 viewDir = normalize(u_Camera.position-v_FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float specularStrength = pow(max(dot(viewDir, reflectDir), 0.0f), u_Material.shininess);
    vec3 specularTexel = texture(u_Material.specularMap, v_TexCoord).rgb;
    vec3 specular = specularStrength*u_Light.specularColor*u_Material.specularColor*specularTexel;

    // Emission
    vec3 emission = texture(u_Material.emissionMap, v_TexCoord).rgb;

    vec3 color = ambient+diffuse+specular+emission;
    f_Color = vec4(color, 1.0);
}

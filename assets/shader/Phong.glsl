#type vertex
#version 460 core
layout(location = 0) in vec3 a_Position;
layout(location = 1) in vec3 a_Normal;
uniform mat4 u_World2Clip;
uniform mat4 u_Model2World;

out vec3 v_Normal;
out vec3 v_FragPos;



void main()
{
    v_Normal = a_Normal;
    v_FragPos = vec3(u_Model2World*vec4(a_Position, 1.0));
    gl_Position = u_World2Clip*vec4(v_FragPos, 1.0);
}

#type fragment
#version 460 core

struct Material
{
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    float shininess;
};

struct Light
{
    vec3 position;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
};

struct Camera
{
    vec3 position;
};



uniform Material u_Material = {vec3(0.1f), vec3(0.6f), vec3(1.0f), 32.0f};
uniform Light u_Light = {vec3(2.0f), vec3(1.0f), vec3(1.0f), vec3(1.0f)};
uniform Camera u_Camera = {vec3(2.0f)};

in vec3 v_Normal;
in vec3 v_FragPos;
out vec4 f_Color;

void main()
{
    // Ambient
    vec3 ambient = u_Light.ambient*u_Material.ambient;

    // Diffuse
    vec3 norm = normalize(v_Normal);
    vec3 lightDir = normalize(u_Light.position-v_FragPos);
    float diff = max(dot(norm, lightDir), 0.0f);
    vec3 diffuse = diff*u_Light.diffuse*u_Material.diffuse;

    // Specular
    vec3 viewDir = normalize(u_Camera.position-v_FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0f), u_Material.shininess);
    vec3 specular = spec*u_Light.specular*u_Material.specular;

    vec3 color = ambient+diffuse+specular;
    f_Color = vec4(color, 1.0);
}

/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : /home/garra/study/dnn/sandbox/layer_learnopengl.h
* author      : Garra
* time        : 2019-10-27 11:06:14
* description : 
*
============================================*/


#pragma once
#include "elsa.h"

class LearnOpenGLLayer : public Layer
{
public:
    LearnOpenGLLayer();

    void OnEvent(Event& e) override;
    void OnUpdate(float deltaTime) override;
    void OnImGuiRender() override;
 
    static std::shared_ptr<LearnOpenGLLayer> Create();

protected:
    void _PrepareSkybox();
    void _PrepareGroundPlane();
    void _PrepareModel();
    void _PrepareUnitCubic();
    void _UpdateMaterialAttributes();

private:
    std::shared_ptr<Viewport> m_viewport = Viewport::Create("LearnOpenGL_Viewport_Main");
    std::shared_ptr<Model> m_crysisNanoSuit = nullptr;
    std::shared_ptr<Model> m_trailer = nullptr;
    std::shared_ptr<Model> m_bulb = nullptr;
    std::shared_ptr<Model> m_handLight = nullptr;
    std::shared_ptr<Shader> m_shaderPos = nullptr;
    std::shared_ptr<Shader> m_shaderColor = nullptr;

    struct 
    {
        glm::vec3* ambientReflectance = nullptr;
        glm::vec3* diffuseReflectance = nullptr;
        glm::vec3* specularReflectance = nullptr;
        glm::vec3* emissiveColor = nullptr;
        float* shininess = nullptr;
        std::shared_ptr<Texture> diffuseMap = nullptr;
        std::shared_ptr<Texture> specularMap = nullptr;
        std::shared_ptr<Texture> emissiveMap = nullptr;
    }
    m_material; 
   
    struct Light
    {
        glm::vec3* color = nullptr;
        glm::vec3* position = nullptr;
    };

    struct DirectionalLight : public Light
    {
        glm::vec3* direction = nullptr;
    }
    m_directionalLight;

    struct PointLight : public Light
    {
        glm::vec3* attenuationCoefficients;
    } 
    m_pointLight;

    struct SpotLight : public PointLight
    {
        glm::vec3* direction = nullptr;
        float innerCone = 6;
        float outerCone = 10;
    }
    m_spotLight, m_flashLight;

    glm::vec3* m_ambientColor = nullptr;

    bool m_showSky = true;
    bool m_showGround= true;

};


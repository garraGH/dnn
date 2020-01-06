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

#include "./learnopengl/pbr.h"
#include "./learnopengl/bloom.h"

class LearnOpenGLLayer : public Layer
{
public:
    LearnOpenGLLayer();

    void OnEvent(Event& e) override;
    void OnUpdate(float deltaTime) override;
    void OnImGuiRender() override;
 
    static std::shared_ptr<LearnOpenGLLayer> Create();


protected:
    void _PrepareModel();
    void _PrepareUniformBuffers();


private:
    std::shared_ptr<Model> m_crysisNanoSuit = nullptr;
    std::shared_ptr<Model> m_silkingMachine = nullptr;
    std::shared_ptr<Model> m_horse = nullptr;
    std::shared_ptr<Model> m_trailer = nullptr;
    std::shared_ptr<Model> m_bulb = nullptr;
    std::shared_ptr<Model> m_handLight = nullptr;
    std::shared_ptr<Model> m_planet = nullptr;
    std::shared_ptr<Model> m_rock = nullptr;

    struct DirectionalLight
    {
        glm::vec3 clr;
        float padding0;
        glm::vec3 dir;
        float intensity;
    };

    struct PointLight
    {
        glm::vec3 clr;
        float padding0;
        glm::vec3 pos;
        float padding1;
        glm::vec3 coe;
        float intensity;
    };

    struct SpotLight
    {
        glm::vec3 clr;
        float cosInnerCone;
        glm::vec3 pos;
        float cosOuterCone;
        glm::vec3 dir;
        float degInnerCone;
        glm::vec3 coe;
        float intensity;
        float degOuterCone;
    };

    DirectionalLight m_dLight = { glm::vec3(1.0f), 0, glm::vec3(0, 0, -1), 1.0f};
    PointLight m_pLight = { glm::vec3(1, 0, 0), 0, glm::vec3(0, 5, 0), 0, glm::vec3(1.0, 0.09, 0.032), 1.0f };
    SpotLight m_sLight = { glm::vec3(0, 1, 0), std::cos(glm::radians(15.0f)), glm::vec3(2, 0, 0), std::cos(glm::radians(20.0f)), glm::vec3(-1, 0, 0), 15, glm::vec3(1.0, 0.22, 0.20), 1.0f, 20 };
    SpotLight m_fLight = { glm::vec3(0, 1, 0), std::cos(glm::radians(15.0f)), glm::vec3(2, 0, 0), std::cos(glm::radians(20.0f)), glm::vec3(-1, 0, 0), 15, glm::vec3(1.0, 0.22, 0.20), 1.0f, 20 };

    const unsigned int m_numOfLights = 2000;

    
    std::shared_ptr<LearnOpenGL> m_current = nullptr;
};


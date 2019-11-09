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
        float* ambient = nullptr;
        float* diffuse = nullptr;
        float* specular = nullptr;
        float* shininess = nullptr;
    }
    m_material; 
   
    struct 
    {
        float* position = nullptr;
        float* ambient = nullptr;
        float* diffuse = nullptr;
        float* specular = nullptr;
    }
    m_light;
};


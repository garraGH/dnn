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
    void _PrepareResources();
    void _PrepareSkybox();
    void _UpdateSkyboxMaterialAttributes();

private:
    std::shared_ptr<Viewport> m_viewport = Viewport::Create("LearnOpenGL_Viewport_Main");
    std::shared_ptr<Model> m_model = nullptr;
    std::shared_ptr<Shader> m_shader = nullptr;

};


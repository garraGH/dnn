/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : sandbox/layer_learnopengl.cpp
* author      : Garra
* time        : 2019-10-27 11:06:13
* description : 
*
============================================*/


#include "layer_learnopengl.h"

std::shared_ptr<LearnOpenGLLayer> LearnOpenGLLayer::Create()
{
    return std::make_shared<LearnOpenGLLayer>();
}

LearnOpenGLLayer::LearnOpenGLLayer()
    : Layer( "LearnOpenGLLayer" )
{
    _PrepareResources();
}

void LearnOpenGLLayer::OnEvent(Event& e)
{
    m_viewport->OnEvent(e);
}

void LearnOpenGLLayer::OnUpdate(float deltaTime)
{
    Renderer::BeginScene(m_viewport);
    m_model->Draw(m_shader);
    Renderer::EndScene();

    m_viewport->OnUpdate(deltaTime);
}

void LearnOpenGLLayer::OnImGuiRender()
{
    m_viewport->OnImGuiRender();
}

void LearnOpenGLLayer::_PrepareResources()
{
    m_model = Model::Create("CysisNanoSuit")->LoadFromFile("/home/garra/study/dnn/assets/mesh/CysisNanoSuit/scene.fbx");
    m_shader = Shader::Create("ModelShader")->LoadFromFile("/home/garra/study/dnn/assets/shader/Model.glsl");
}


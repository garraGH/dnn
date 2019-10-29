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
#include "glm/gtc/type_ptr.hpp"

std::shared_ptr<LearnOpenGLLayer> LearnOpenGLLayer::Create()
{
    return std::make_shared<LearnOpenGLLayer>();
}

LearnOpenGLLayer::LearnOpenGLLayer()
    : Layer( "LearnOpenGLLayer" )
{
    _PrepareResources();
    _PrepareSkybox();
}

void LearnOpenGLLayer::OnEvent(Event& e)
{
    m_viewport->OnEvent(e);
}

void LearnOpenGLLayer::OnUpdate(float deltaTime)
{
    _UpdateMaterialAttribute();

    Renderer::BeginScene(m_viewport);
    m_model->Draw(m_shader);
    Renderer::Submit("Skybox", "Skybox");
    Renderer::EndScene();

    m_viewport->OnUpdate(deltaTime);
}

void LearnOpenGLLayer::OnImGuiRender()
{
    m_viewport->OnImGuiRender();
}

void LearnOpenGLLayer::_PrepareResources()
{
//     m_model = Model::Create("CysisNanoSuit")->LoadFromFile("/home/garra/study/dnn/assets/mesh/CysisNanoSuit/scene.fbx");
    m_model = Model::Create("trailer")->LoadFromFile("/home/garra/study/dnn/assets/mesh/trailer/Alena_Shek.obj");
    m_shader = Shader::Create("ModelShader")->LoadFromFile("/home/garra/study/dnn/assets/shader/Model.glsl");

    auto [mMin, mMax] = m_model->GetAABB();
    std::shared_ptr<Camera> cam = m_viewport->GetCamera();
    float dx = mMax.x-mMin.x;
    float dy = mMax.y-mMin.y;
    float dw = 2.0*sqrt(dx*dx+dy*dy);
    cam->SetWidth(dw);
    cam->SetPosition(glm::vec3(0, 0, dw*2));
}

void LearnOpenGLLayer::_PrepareSkybox()
{
    float x = glm::tan(glm::radians(22.5));
    float vertices[] = 
    { 
        -1, -1, -x, +x, 1, 
        -1, +1, -x, -x, 1, 
        +1, +1, +x, -x, 1, 
        +1, -1, +x, +x, 1, 
    };
    unsigned char indices[] = { 0, 1, 2, 0, 2, 3 };
    Buffer::Layout layoutVextex = 
    {
        {Buffer::Element::DataType::Float2, "a_Position", false}, 
        {Buffer::Element::DataType::Float3, "a_Direction", false}
    };
    Buffer::Layout layoutIndex = { {Buffer::Element::DataType::UChar} };
    std::shared_ptr<Buffer> vb = Buffer::CreateVertex(sizeof(vertices), vertices)->SetLayout(layoutVextex);
    std::shared_ptr<Buffer> ib = Buffer::CreateIndex(sizeof(indices), indices)->SetLayout(layoutIndex);
    std::shared_ptr<Elsa::Mesh> mesh= Renderer::Resources::Create<Elsa::Mesh>("Skybox")->Set(ib, {vb});
    INFO("meshptr:{}, {}", mesh->GetBufferArray(), (void*)(mesh->GetBufferArray().get()));
    std::shared_ptr<Texture> tex = Renderer::Resources::Create<Texture2D>("Skybox")->LoadFromFile("/home/garra/study/dnn/assets/texture/skybox/autumn-crossing_3.jpg");
    std::shared_ptr<Material::Attribute> aCameraDirection = Renderer::Resources::Create<Material::Attribute>("CameraDirection")->SetType(Material::Attribute::Type::Float3);
    std::shared_ptr<Material> mtr = Renderer::Resources::Create<Material>("Skybox")->AddTexture("u_Skybox", tex)->Set("u_CameraDirection", aCameraDirection);
    Renderer::Resources::Create<Renderer::Element>("Skybox")->Set(mesh, mtr);
    Renderer::Resources::Create<Shader>("Skybox")->LoadFromFile("/home/garra/study/dnn/assets/shader/Skybox.glsl");
}

void LearnOpenGLLayer::_UpdateMaterialAttribute()
{   
    Renderer::Resources::Get<Material::Attribute>("CameraDirection")->UpdateData(glm::value_ptr(m_viewport->GetCamera()->GetDirection()));
}

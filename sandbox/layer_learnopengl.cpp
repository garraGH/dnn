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
    _PrepareSkybox();
    _PrepareUnitCubic();
    _PrepareGroundPlane();
    _PrepareModel();
}

void LearnOpenGLLayer::OnEvent(Event& e)
{
    m_viewport->OnEvent(e);
}

void LearnOpenGLLayer::OnUpdate(float deltaTime)
{
    _UpdateMaterialAttributes();

    Renderer::BeginScene(m_viewport);
    m_crysisNanoSuit->Draw(m_shaderPos);
//     m_bulb->Draw(m_shaderColor);
    m_handLight->Draw(m_shaderColor);
    Renderer::Submit("GroundPlane", "GroundPlane");
    Renderer::Submit("Skybox", "Skybox");
    Renderer::Submit("UnitCubic", "Phong");
    Renderer::EndScene();

    m_viewport->OnUpdate(deltaTime);
}

void LearnOpenGLLayer::OnImGuiRender()
{
    m_viewport->OnImGuiRender();
    ImGui::ColorPicker3("lightcolor", m_lightColor);
    ImGui::ColorPicker3("objectColor", m_objectColor);
}

void LearnOpenGLLayer::_PrepareModel()
{
    m_crysisNanoSuit = Renderer::Resources::Create<Model>("CysisNanoSuit")->LoadFromFile("/home/garra/study/dnn/assets/mesh/CysisNanoSuit/scene.fbx");
    m_trailer = Renderer::Resources::Create<Model>("Trailer")->LoadFromFile("/home/garra/study/dnn/assets/mesh/Trailer/Alena_Shek.obj");
    m_bulb = Renderer::Resources::Create<Model>("Bulb")->LoadFromFile("/home/garra/study/dnn/assets/mesh/Bulb/Bulbs.3ds");
    m_handLight = Renderer::Resources::Create<Model>("HandLight")->LoadFromFile("/home/garra/study/dnn/assets/mesh/HandLight/hand_light.blend");
    m_shaderPos = Renderer::Resources::Create<Shader>("Pos")->LoadFromFile("/home/garra/study/dnn/assets/shader/Model.glsl");
    m_shaderColor = Renderer::Resources::Create<Shader>("Phong")->LoadFromFile("/home/garra/study/dnn/assets/shader/Phong.glsl");

//     m_viewport->GetCamera()->SetPosition(glm::vec3(0, 50, 50));
//     auto [mMin, mMax] = m_model->GetAABB();
//     std::shared_ptr<Camera> cam = m_viewport->GetCamera();
//     float dx = mMax.x-mMin.x;
//     float dy = mMax.y-mMin.y;
//     float dw = 2.0*sqrt(dx*dx+dy*dy);
//     cam->SetWidth(dw);
//     cam->SetPosition(glm::vec3(0, dy*2, dw*2));
}

void LearnOpenGLLayer::_PrepareUnitCubic()
{
    float vertices[] = 
    {
        -1, -1, +1, 
        +1, -1, +1, 
        +1, +1, +1, 
        -1, +1, +1, 
        -1, -1, -1, 
        +1, -1, -1, 
        +1, +1, -1, 
        -1, +1, -1, 
    };

    unsigned char indices[] = 
    { 
        0, 1, 2, 
        0, 2, 3, 
        1, 5, 6, 
        1, 6, 2, 
        4, 6, 5, 
        4, 7, 6, 
        0, 3, 7, 
        0, 7, 4, 
        0, 4, 5, 
        0, 5, 1, 
        3, 2, 6, 
        3, 6, 7
    };
    Buffer::Layout layoutVextex = { {Buffer::Element::DataType::Float3, "a_Position", false} };
    Buffer::Layout layoutIndex = { {Buffer::Element::DataType::UChar} };
    std::shared_ptr<Buffer> vb = Buffer::CreateVertex(sizeof(vertices), vertices)->SetLayout(layoutVextex);
    std::shared_ptr<Buffer> ib = Buffer::CreateIndex(sizeof(indices), indices)->SetLayout(layoutIndex);
    std::shared_ptr<Elsa::Mesh> mesh= Renderer::Resources::Create<Elsa::Mesh>("UnitCubic")->Set(ib, {vb});
    using MA = Material::Attribute;
    std::shared_ptr<MA> maLightColor = Renderer::Resources::Create<MA>("LightColor")->SetType(MA::Type::Float3);
    std::shared_ptr<MA> maObjectColor = Renderer::Resources::Create<MA>("ObjectColor")->SetType(MA::Type::Float3);
    m_lightColor = (float*)maLightColor->GetData();
    m_objectColor = (float*)maObjectColor->GetData();
    std::shared_ptr<Material> mtr = Renderer::Resources::Create<Material>("UnitCubic")->Set("u_LightColor", maLightColor)->Set("u_ObjectColor", maObjectColor);
    Renderer::Resources::Create<Shader>("Default")->LoadFromFile("/home/garra/study/dnn/assets/shader/Default.glsl");
    Renderer::Resources::Create<Renderer::Element>("UnitCubic")->Set(mesh, mtr);

}

void LearnOpenGLLayer::_PrepareSkybox()
{
    float vertices[] = { -1, -1, +1, -1, +1, +1, -1, +1 };
    unsigned char indices[] = { 0, 1, 2, 0, 2, 3 };
    Buffer::Layout layoutVextex = { {Buffer::Element::DataType::Float2, "a_Position", false} };
    Buffer::Layout layoutIndex = { {Buffer::Element::DataType::UChar} };
    std::shared_ptr<Buffer> vb = Buffer::CreateVertex(sizeof(vertices), vertices)->SetLayout(layoutVextex);
    std::shared_ptr<Buffer> ib = Buffer::CreateIndex(sizeof(indices), indices)->SetLayout(layoutIndex);
    std::shared_ptr<Elsa::Mesh> mesh= Renderer::Resources::Create<Elsa::Mesh>("BackgroundPlane")->Set(ib, {vb});
    std::shared_ptr<Texture> tex = Renderer::Resources::Create<Texture2D>("Skybox");
    tex->LoadFromFile("/home/garra/study/dnn/assets/texture/skybox/autumn-crossing_3.jpg");
    using MA = Material::Attribute;
    std::shared_ptr<MA> maNearCorners = Renderer::Resources::Create<MA>("NearCorners")->Set(MA::Type::Float3, 4);
    std::shared_ptr<MA> maFarCorners = Renderer::Resources::Create<MA>("FarCorners")->Set(MA::Type::Float3, 4);
    std::shared_ptr<MA> maCornersDirection = Renderer::Resources::Create<MA>("CornersDirection")->Set(MA::Type::Float3, 4);
    std::shared_ptr<Material> mtr = Renderer::Resources::Create<Material>("Skybox")->AddTexture("u_Skybox", tex);
    mtr->Set("u_NearCorners", maNearCorners)->Set("u_FarCorners", maFarCorners);
    Renderer::Resources::Create<Renderer::Element>("Skybox")->Set(mesh, mtr);
    Renderer::Resources::Create<Shader>("Skybox")->LoadFromFile("/home/garra/study/dnn/assets/shader/Skybox.glsl");
}

void LearnOpenGLLayer::_PrepareGroundPlane()
{
    using MA = Material::Attribute;
    std::shared_ptr<Material> mtr = Renderer::Resources::Create<Material>("GroundPlane");
    mtr->Set("u_NearCorners", Renderer::Resources::Get<MA>("NearCorners"));
    mtr->Set("u_FarCorners", Renderer::Resources::Get<MA>("FarCorners"));
    Renderer::Resources::Create<Renderer::Element>("GroundPlane")->SetMesh("BackgroundPlane")->SetMaterial(mtr);
    Renderer::Resources::Create<Shader>("GroundPlane")->LoadFromFile("/home/garra/study/dnn/assets/shader/GroundPlane.glsl");
}

void LearnOpenGLLayer::_UpdateMaterialAttributes()
{
    Renderer::Resources::Get<Material::Attribute>("NearCorners")->UpdateData(&m_viewport->GetCamera()->GetNearCornersInWorldSpace()[0]);
    Renderer::Resources::Get<Material::Attribute>("FarCorners")->UpdateData(&m_viewport->GetCamera()->GetFarCornersInWorldSpace()[0]);
}

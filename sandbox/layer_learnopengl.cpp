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
    ImGui::PushItemWidth(120);
    if(ImGui::CollapsingHeader("Material"))
    {
        ImGui::ColorPicker3("AmbientColor", m_material.ambientColor);
        ImGui::ColorPicker3("DiffuseColor", m_material.diffuseColor);
        ImGui::ColorPicker3("SpecularColor", m_material.specularColor);
        ImGui::SliderFloat("Shininess", m_material.shininess, 0, 512);
    }

    if(ImGui::CollapsingHeader("Light"))
    {
        ImGui::ColorPicker3("AmbientColor", m_light.ambientColor);
        ImGui::ColorPicker3("DiffuseColor", m_light.diffuseColor);
        ImGui::ColorPicker3("SpecularColor", m_light.specularColor);
        ImGui::SliderFloat3("Position", m_light.position, -5, 5);
    }
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
        // front
        -1, -1, +1, 0, 0, +1, 0, 0,   
        +1, -1, +1, 0, 0, +1, 1, 0, 
        +1, +1, +1, 0, 0, +1, 1, 1, 
        -1, +1, +1, 0, 0, +1, 0, 1, 
        // back
        +1, -1, -1, 0, 0, -1, 0, 0, 
        -1, -1, -1, 0, 0, -1, 1, 0, 
        -1, +1, -1, 0, 0, -1, 1, 1, 
        +1, +1, -1, 0, 0, -1, 0, 1, 
        // left
        -1, -1, -1, -1, 0, 0, 0, 0, 
        -1, -1, +1, -1, 0, 0, 1, 0, 
        -1, +1, +1, -1, 0, 0, 1, 1, 
        -1, +1, -1, -1, 0, 0, 0, 1, 
        // right
        +1, -1, +1, +1, 0, 0, 0, 0, 
        +1, -1, -1, +1, 0, 0, 1, 0, 
        +1, +1, -1, +1, 0, 0, 1, 1, 
        +1, +1, +1, +1, 0, 0, 0, 1, 
        // up
        -1, +1, +1, 0, +1, 0, 0, 0, 
        +1, +1, +1, 0, +1, 0, 1, 0, 
        +1, +1, -1, 0, +1, 0, 1, 1, 
        -1, +1, -1, 0, +1, 0, 0, 1, 
        // down
        -1, -1, -1, 0, -1, 0, 0, 0, 
        +1, -1, -1, 0, -1, 0, 1, 0, 
        +1, -1, +1, 0, -1, 0, 1, 1, 
        -1, -1, +1, 0, -1, 0, 0, 1, 
    };

    unsigned char indices[] = 
    { 
        0, 1, 2, 
        0, 2, 3, 
        4, 5, 6, 
        4, 6, 7, 
        8, 9, 10, 
        8, 10, 11, 
        12, 13, 14, 
        12, 14, 15, 
        16, 17, 18, 
        16, 18, 19, 
        20, 21, 22, 
        20, 22, 23
    };
    Buffer::Layout layoutVextex = { {Buffer::Element::DataType::Float3, "a_Position", false}, 
                                    {Buffer::Element::DataType::Float3, "a_Normal", false},
                                    {Buffer::Element::DataType::Float2, "a_TexCoord", false}  };
    Buffer::Layout layoutIndex = { {Buffer::Element::DataType::UChar} };
    std::shared_ptr<Buffer> vb = Buffer::CreateVertex(sizeof(vertices), vertices)->SetLayout(layoutVextex);
    std::shared_ptr<Buffer> ib = Buffer::CreateIndex(sizeof(indices), indices)->SetLayout(layoutIndex);
    std::shared_ptr<Elsa::Mesh> mesh= Renderer::Resources::Create<Elsa::Mesh>("UnitCubic")->Set(ib, {vb});
    using MA = Material::Attribute;

    std::shared_ptr<MA> maLightPosition = Renderer::Resources::Create<MA>("LightPosition")->Set(MA::Type::Float3, 1, glm::value_ptr(glm::vec3(0, 2, 0)));
    std::shared_ptr<MA> maLightAmbientColor = Renderer::Resources::Create<MA>("LightAmbientColor")->Set(MA::Type::Float3, 1, glm::value_ptr(glm::vec3(1.0f)));
    std::shared_ptr<MA> maLightDiffuseColor = Renderer::Resources::Create<MA>("LightDiffuseColor")->Set(MA::Type::Float3, 1, glm::value_ptr(glm::vec3(1.0f)));
    std::shared_ptr<MA> maLightSpecularColor = Renderer::Resources::Create<MA>("LightSpecularColor")->Set(MA::Type::Float3, 1, glm::value_ptr(glm::vec3(1.0f)));

    std::shared_ptr<MA> maMaterialAmbientColor = Renderer::Resources::Create<MA>("MaterialAmbientColor")->Set(MA::Type::Float3, 1, glm::value_ptr(glm::vec3(0.1f)));
    std::shared_ptr<MA> maMaterialDiffuseColor = Renderer::Resources::Create<MA>("MaterialDiffuseColor")->Set(MA::Type::Float3, 1, glm::value_ptr(glm::vec3(0.5f)));
    std::shared_ptr<MA> maMaterialSpecularColor = Renderer::Resources::Create<MA>("MaterialSpecularColor")->Set(MA::Type::Float3, 1, glm::value_ptr(glm::vec3(1.0f)));
    std::shared_ptr<MA> maMaterialShininess = Renderer::Resources::Create<MA>("MaterialShininess")->SetType(MA::Type::Float1);

    std::shared_ptr<MA> maCameraPosition = Renderer::Resources::Create<MA>("CameraPosition")->Set(MA::Type::Float3, 1, glm::value_ptr(glm::vec3(2.0f)));


    m_material.ambientColor = (float*)maMaterialAmbientColor->GetData();
    m_material.diffuseColor = (float*)maMaterialDiffuseColor->GetData();
    m_material.specularColor = (float*)maMaterialSpecularColor->GetData();
    m_material.shininess = (float*)maMaterialShininess->GetData();
    m_material.diffuseMap = Renderer::Resources::Create<Texture2D>("DiffuseMap")->LoadFromFile("/home/garra/study/dnn/assets/texture/container2.png");
    m_material.specularMap = Renderer::Resources::Create<Texture2D>("SpecularMap")->LoadFromFile("/home/garra/study/dnn/assets/texture/lighting_maps_specular_color.png");
    m_material.emissionMap = Renderer::Resources::Create<Texture2D>("EmissionMap")->LoadFromFile("/home/garra/study/dnn/assets/texture/matrix.jpg");
    *m_material.shininess = 32.0f;

    m_light.ambientColor = (float*)maLightAmbientColor->GetData();
    m_light.diffuseColor = (float*)maLightDiffuseColor->GetData();
    m_light.specularColor = (float*)maLightSpecularColor->GetData();
    m_light.position = (float*)maLightPosition->GetData();


    std::shared_ptr<Material> mtr = Renderer::Resources::Create<Material>("UnitCubic");
    mtr->Set("u_Material.ambientColor", maMaterialAmbientColor);
    mtr->Set("u_Material.diffuseColor", maMaterialDiffuseColor);
    mtr->Set("u_Material.specularColor", maMaterialSpecularColor);
    mtr->Set("u_Material.shininess", maMaterialShininess);
    mtr->AddTexture("u_Material.diffuseMap", m_material.diffuseMap);
    mtr->AddTexture("u_Material.specularMap", m_material.specularMap);
    mtr->AddTexture("u_Material.emissionMap", m_material.emissionMap);
    mtr->Set("u_Light.position", maLightPosition);
    mtr->Set("u_Light.ambientColor", maLightAmbientColor);
    mtr->Set("u_Light.diffuseColor", maLightDiffuseColor);
    mtr->Set("u_Light.specularColor", maLightSpecularColor);
    mtr->Set("u_Camera.position", maCameraPosition);

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
    Renderer::Resources::Get<Material::Attribute>("CameraPosition")->UpdateData(&m_viewport->GetCamera()->GetPosition());
    Renderer::Resources::Get<Material::Attribute>("NearCorners")->UpdateData(&m_viewport->GetCamera()->GetNearCornersInWorldSpace()[0]);
    Renderer::Resources::Get<Material::Attribute>("FarCorners")->UpdateData(&m_viewport->GetCamera()->GetFarCornersInWorldSpace()[0]);
}

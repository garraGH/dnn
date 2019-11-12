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
    Renderer::Submit("UnitCubic", "Blinn-Phong", 100);
    Renderer::EndScene();

    m_viewport->OnUpdate(deltaTime);
}

void LearnOpenGLLayer::OnImGuiRender()
{
    using MA = Material::Attribute;
    m_viewport->OnImGuiRender();
    ImGui::PushItemWidth(120);
    if(ImGui::CollapsingHeader("Environment"))
    {
        ImGui::ColorPicker3("AmbientColor", reinterpret_cast<float*>(m_ambientColor));
    }

    if(ImGui::CollapsingHeader("Material"))
    {
        ImGui::ColorPicker3("AmbientReflectance", reinterpret_cast<float*>(m_material.ambientReflectance));
        ImGui::ColorPicker3("DiffuseReflectance", reinterpret_cast<float*>(m_material.diffuseReflectance));
        ImGui::ColorPicker3("SpecularReflectance", reinterpret_cast<float*>(m_material.specularReflectance));
        ImGui::ColorPicker3("EmissiveColor", reinterpret_cast<float*>(m_material.emissiveColor));
        ImGui::SliderFloat("Shininess", m_material.shininess, 0, 512);
    }

    if(ImGui::CollapsingHeader("Light"))
    {
        ImGui::SetNextWindowPos({100, 0});
        if(ImGui::CollapsingHeader("DirectionalLight"))
        {
            ImGui::ColorPicker3("Color", reinterpret_cast<float*>(m_directionalLight.color));
            ImGui::SliderFloat3("Direction", reinterpret_cast<float*>(m_directionalLight.direction), -1, 1);
        }
        if(ImGui::CollapsingHeader("PointLight"))
        {
            ImGui::ColorPicker3("Color", reinterpret_cast<float*>(m_pointLight.color));
            ImGui::SliderFloat3("Position", reinterpret_cast<float*>(m_pointLight.position), -10, 10);
            ImGui::SliderFloat3("AttenuationCoefficents", reinterpret_cast<float*>(m_pointLight.attenuationCoefficients), -10, 10);
        }
        if(ImGui::CollapsingHeader("SpotLight"))
        {
            ImGui::ColorPicker3("Color", reinterpret_cast<float*>(m_spotLight.color));
            ImGui::SliderFloat3("Position", reinterpret_cast<float*>(m_spotLight.position), -10, 10);
            ImGui::SliderFloat3("AttenuationCoefficents", reinterpret_cast<float*>(m_spotLight.attenuationCoefficients), -10, 10);
            ImGui::SliderFloat3("Direction", reinterpret_cast<float*>(m_spotLight.direction), -1, 1);
            if(ImGui::SliderFloat("InnerCone", &m_spotLight.innerCone, 0, 90))
            {
                float innerCone = std::cos(glm::radians(m_spotLight.innerCone));
                Renderer::Resources::Get<MA>("SLightInnerCone")->UpdateData(&innerCone);
            }
            if(ImGui::SliderFloat("OuterCone", &m_spotLight.outerCone, 0, 120))
            {
                float outerCone = std::cos(glm::radians(m_spotLight.outerCone));
                Renderer::Resources::Get<MA>("SLightOuterCone")->UpdateData(&outerCone);
            }
        }
        if(ImGui::CollapsingHeader("FlashLight"))
        {
            ImGui::ColorPicker3("Color", reinterpret_cast<float*>(m_flashLight.color));
            ImGui::SliderFloat3("AttenuationCoefficents", reinterpret_cast<float*>(m_flashLight.attenuationCoefficients), -10, 10);
            if(ImGui::SliderFloat("InnerCone", &m_flashLight.innerCone, 0, 90))
            {
                float innerCone = std::cos(glm::radians(m_flashLight.innerCone));
                Renderer::Resources::Get<MA>("FLightInnerCone")->UpdateData(&innerCone);
            }
            if(ImGui::SliderFloat("OuterCone", &m_flashLight.outerCone, 0, 120))
            {
                float outerCone = std::cos(glm::radians(m_flashLight.outerCone));
                Renderer::Resources::Get<MA>("FLightOuterCone")->UpdateData(&outerCone);
            }
        }
    }
}

void LearnOpenGLLayer::_PrepareModel()
{
    m_crysisNanoSuit = Renderer::Resources::Create<Model>("CysisNanoSuit")->LoadFromFile("/home/garra/study/dnn/assets/mesh/CysisNanoSuit/scene.fbx");
    m_trailer = Renderer::Resources::Create<Model>("Trailer")->LoadFromFile("/home/garra/study/dnn/assets/mesh/Trailer/Alena_Shek.obj");
    m_bulb = Renderer::Resources::Create<Model>("Bulb")->LoadFromFile("/home/garra/study/dnn/assets/mesh/Bulb/Bulbs.3ds");
    m_handLight = Renderer::Resources::Create<Model>("HandLight")->LoadFromFile("/home/garra/study/dnn/assets/mesh/HandLight/hand_light.blend");
    m_shaderPos = Renderer::Resources::Create<Shader>("Pos")->LoadFromFile("/home/garra/study/dnn/assets/shader/Model.glsl");
    m_shaderColor = Renderer::Resources::Create<Shader>("Blinn-Phong")->LoadFromFile("/home/garra/study/dnn/assets/shader/Blinn-Phong.glsl");

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
    glm::vec3 displacements[100];
    int k = 0;
    for(float i=-20; i<20; i+=4)
    {
        for(float j=-20; j<20; j+=4)
        {
            displacements[k++] = glm::vec3(i, 1, j);
        }
    }

    Buffer::Layout layoutVextex = { {Buffer::Element::DataType::Float3, "a_Position", false}, 
                                    {Buffer::Element::DataType::Float3, "a_Normal", false},
                                    {Buffer::Element::DataType::Float2, "a_TexCoord", false}  };
    Buffer::Layout layoutIndex = { {Buffer::Element::DataType::UChar} };
    Buffer::Layout layoutInstance = { {Buffer::Element::DataType::Float3, "a_Displacement", false, 1} };

    std::shared_ptr<Buffer> vertexBuffer = Buffer::CreateVertex(sizeof(vertices), vertices)->SetLayout(layoutVextex);
    std::shared_ptr<Buffer> indexBuffer = Buffer::CreateIndex(sizeof(indices), indices)->SetLayout(layoutIndex);
    std::shared_ptr<Buffer> instanceBuffer = Buffer::CreateVertex(100*sizeof(glm::vec3), glm::value_ptr(displacements[0]))->SetLayout(layoutInstance);
    std::shared_ptr<Elsa::Mesh> mesh= Renderer::Resources::Create<Elsa::Mesh>("UnitCubic")->Set(indexBuffer, {vertexBuffer, instanceBuffer});
    using MA = Material::Attribute;


    std::shared_ptr<MA> maMaterialAmbientReflectance = Renderer::Resources::Create<MA>("MaterialAmbientReflectance")->Set(MA::Type::Float3, 1, glm::value_ptr(glm::vec3(0.1f)));
    std::shared_ptr<MA> maMaterialDiffuseReflectance = Renderer::Resources::Create<MA>("MaterialDiffuseReflectance")->Set(MA::Type::Float3, 1, glm::value_ptr(glm::vec3(0.5f)));
    std::shared_ptr<MA> maMaterialSpecularReflectance = Renderer::Resources::Create<MA>("MaterialSpecularReflectance")->Set(MA::Type::Float3, 1, glm::value_ptr(glm::vec3(1.0f)));
    std::shared_ptr<MA> maMaterialEmissiveColor = Renderer::Resources::Create<MA>("MaterialEmissiveColor")->Set(MA::Type::Float3, 1, glm::value_ptr(glm::vec3(0.1f)));
    std::shared_ptr<MA> maMaterialShininess = Renderer::Resources::Create<MA>("MaterialShininess")->SetType(MA::Type::Float1);

    std::shared_ptr<MA> maCameraPosition = Renderer::Resources::Create<MA>("CameraPosition")->Set(MA::Type::Float3, 1, glm::value_ptr(glm::vec3(2.0f)));


    m_material.ambientReflectance = reinterpret_cast<glm::vec3*>(maMaterialAmbientReflectance->GetData());
    m_material.diffuseReflectance = reinterpret_cast<glm::vec3*>(maMaterialDiffuseReflectance->GetData());
    m_material.specularReflectance = reinterpret_cast<glm::vec3*>(maMaterialSpecularReflectance->GetData());
    m_material.emissiveColor = reinterpret_cast<glm::vec3*>(maMaterialEmissiveColor->GetData());
    m_material.shininess = reinterpret_cast<float*>(maMaterialShininess->GetData());
    m_material.diffuseMap = Renderer::Resources::Create<Texture2D>("DiffuseMap")->LoadFromFile("/home/garra/study/dnn/assets/texture/container2.png");
    m_material.specularMap = Renderer::Resources::Create<Texture2D>("SpecularMap")->LoadFromFile("/home/garra/study/dnn/assets/texture/lighting_maps_specular_color.png");
    m_material.emissiveMap = Renderer::Resources::Create<Texture2D>("EmissiveMap")->LoadFromFile("/home/garra/study/dnn/assets/texture/matrix.jpg");
    *m_material.shininess = 32.0f;


    // AmbientColor
    std::shared_ptr<MA> maAmbientColor = Renderer::Resources::Create<MA>("AmbientColor")->Set(MA::Type::Float3, 1, glm::value_ptr(glm::vec3(0.1f)));
    m_ambientColor = reinterpret_cast<glm::vec3*>(maAmbientColor->GetData());

    // DirectionalLight
    std::shared_ptr<MA> maDLightColor = Renderer::Resources::Create<MA>("DLightColor")->Set(MA::Type::Float3, 1, glm::value_ptr(glm::vec3(1.0f)));
    std::shared_ptr<MA> maDLightDirection = Renderer::Resources::Create<MA>("DLightDirection")->Set(MA::Type::Float3, 1, glm::value_ptr(glm::vec3(-1, -2, -3)));
    m_directionalLight.color = reinterpret_cast<glm::vec3*>(maDLightColor->GetData());
    m_directionalLight.direction = reinterpret_cast<glm::vec3*>(maDLightDirection->GetData());

    // PointLight
    std::shared_ptr<MA> maPLightColor = Renderer::Resources::Create<MA>("PLightColor")->Set(MA::Type::Float3, 1, glm::value_ptr(glm::vec3(0.1, 0.8, 0.2)));
    std::shared_ptr<MA> maPLightPosition = Renderer::Resources::Create<MA>("PLightPosition")->Set(MA::Type::Float3, 1, glm::value_ptr(glm::vec3(0, 2, 0)));
    std::shared_ptr<MA> maPLightCoefs = Renderer::Resources::Create<MA>("PLightCoefs")->Set(MA::Type::Float3, 1, glm::value_ptr(glm::vec3(1.0, 1.0, 2.0)));
    m_pointLight.color = reinterpret_cast<glm::vec3*>(maPLightColor->GetData());
    m_pointLight.position = reinterpret_cast<glm::vec3*>(maPLightPosition->GetData());
    m_pointLight.attenuationCoefficients = reinterpret_cast<glm::vec3*>(maPLightCoefs->GetData());

    // SpotLight
    std::shared_ptr<MA> maSLightColor = Renderer::Resources::Create<MA>("SLightColor")->Set(MA::Type::Float3, 1, glm::value_ptr(glm::vec3(0.1, 0.1, 0.9)));
    std::shared_ptr<MA> maSLightPosition = Renderer::Resources::Create<MA>("SLightPosition")->Set(MA::Type::Float3, 1, glm::value_ptr(glm::vec3(0, 0, 2)));
    std::shared_ptr<MA> maSLightCoefs = Renderer::Resources::Create<MA>("SLightCoefs")->Set(MA::Type::Float3, 1, glm::value_ptr(glm::vec3(1.0, 1.0, 2.0)));
    std::shared_ptr<MA> maSLightDirection = Renderer::Resources::Create<MA>("SLightDirection")->Set(MA::Type::Float3, 1, glm::value_ptr(glm::vec3(0, 0, -1)));
    std::shared_ptr<MA> maSLightInnerCone = Renderer::Resources::Create<MA>("SLightInnerCone")->Set(MA::Type::Float1, 1, &m_spotLight.innerCone);
    std::shared_ptr<MA> maSLightOuterCone = Renderer::Resources::Create<MA>("SLightOuterCone")->Set(MA::Type::Float1, 1, &m_spotLight.outerCone);
    m_spotLight.color = reinterpret_cast<glm::vec3*>(maSLightColor->GetData());
    m_spotLight.position = reinterpret_cast<glm::vec3*>(maSLightPosition->GetData());
    m_spotLight.attenuationCoefficients = reinterpret_cast<glm::vec3*>(maSLightCoefs->GetData());
    m_spotLight.direction = reinterpret_cast<glm::vec3*>(maSLightDirection->GetData());

    // FlashLight
    std::shared_ptr<MA> maFLightColor = Renderer::Resources::Create<MA>("FLightColor")->Set(MA::Type::Float3, 1, glm::value_ptr(glm::vec3(0.8, 0.1, 0.2)));
    std::shared_ptr<MA> maFLightPosition = Renderer::Resources::Create<MA>("FLightPosition")->Set(MA::Type::Float3, 1, glm::value_ptr(m_viewport->GetCamera()->GetPosition()));
    std::shared_ptr<MA> maFLightCoefs = Renderer::Resources::Create<MA>("FLightCoefs")->Set(MA::Type::Float3, 1, glm::value_ptr(glm::vec3(1.0, 1.0, 2.0)));
    std::shared_ptr<MA> maFLightDirection = Renderer::Resources::Create<MA>("FLightDirection")->Set(MA::Type::Float3, 1, glm::value_ptr(m_viewport->GetCamera()->GetDirection()));
    std::shared_ptr<MA> maFLightInnerCone = Renderer::Resources::Create<MA>("FLightInnerCone")->Set(MA::Type::Float1, 1, &m_flashLight.innerCone);
    std::shared_ptr<MA> maFLightOuterCone = Renderer::Resources::Create<MA>("FLightOuterCone")->Set(MA::Type::Float1, 1, &m_flashLight.outerCone);
    m_flashLight.color = reinterpret_cast<glm::vec3*>(maFLightColor->GetData());
    m_flashLight.attenuationCoefficients = reinterpret_cast<glm::vec3*>(maFLightCoefs->GetData());

    std::shared_ptr<Material> mtr = Renderer::Resources::Create<Material>("UnitCubic");
    mtr->Set("u_Material.ambientReflectance", maMaterialAmbientReflectance);
    mtr->Set("u_Material.diffuseReflectance", maMaterialDiffuseReflectance);
    mtr->Set("u_Material.specularReflectance", maMaterialSpecularReflectance);
    mtr->Set("u_Material.emissiveColor", maMaterialEmissiveColor);
    mtr->Set("u_Material.shininess", maMaterialShininess);
    mtr->AddTexture("u_Material.diffuseMap", m_material.diffuseMap);
    mtr->AddTexture("u_Material.specularMap", m_material.specularMap);
    mtr->AddTexture("u_Material.emissiveMap", m_material.emissiveMap);

    // AmbientColor
    mtr->Set("u_AmbientColor", maAmbientColor);

    // DirectionalLight
    mtr->Set("u_DirectionalLight[0].color", maDLightColor);
    mtr->Set("u_DirectionalLight[0].direction", maDLightDirection);

    // PointLight
    mtr->Set("u_PointLight[0].position", maPLightPosition);
    mtr->Set("u_PointLight[0].color", maPLightColor);
    mtr->Set("u_PointLight[0].attenuationCoefficients", maPLightCoefs);

    // SpotLight
    mtr->Set("u_SpotLight[0].position", maSLightPosition);
    mtr->Set("u_SpotLight[0].direction", maSLightDirection);
    mtr->Set("u_SpotLight[0].attenuationCoefficients", maSLightCoefs);
    mtr->Set("u_SpotLight[0].color", maSLightColor);
    mtr->Set("u_SpotLight[0].innerCone", maSLightInnerCone);
    mtr->Set("u_SpotLight[0].outerCone", maSLightOuterCone);
    
    // FlashLight
    mtr->Set("u_FlashLight.position", maFLightPosition);
    mtr->Set("u_FlashLight.direction", maFLightDirection);
    mtr->Set("u_FlashLight.attenuationCoefficients", maFLightCoefs);
    mtr->Set("u_FlashLight.color", maFLightColor);
    mtr->Set("u_FlashLight.innerCone", maFLightInnerCone);
    mtr->Set("u_FlashLight.outerCone", maFLightOuterCone);

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
    using MA = Material::Attribute;
    const std::shared_ptr<Camera>& cam = m_viewport->GetCamera();
    Renderer::Resources::Get<MA>("CameraPosition")->UpdateData(&cam->GetPosition());
    Renderer::Resources::Get<MA>("NearCorners")->UpdateData(&cam->GetNearCornersInWorldSpace()[0]);
    Renderer::Resources::Get<MA>("FarCorners")->UpdateData(&cam->GetFarCornersInWorldSpace()[0]);
    Renderer::Resources::Get<MA>("FLightPosition")->UpdateData(&cam->GetPosition());
    glm::vec3 dir = cam->GetDirection();
    Renderer::Resources::Get<MA>("FLightDirection")->UpdateData(&dir);
}


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
#include "glm/gtx/string_cast.hpp"
#include "osdialog.h"

std::shared_ptr<LearnOpenGLLayer> LearnOpenGLLayer::Create()
{
    return std::make_shared<LearnOpenGLLayer>();
}

LearnOpenGLLayer::LearnOpenGLLayer()
    : Layer( "LearnOpenGLLayer" )
{

    _PrepareUniformBuffers();
    m_current = std::make_shared<PBR>();
}

void LearnOpenGLLayer::OnEvent(Event& e)
{
    m_current->OnEvent(e);
}

void LearnOpenGLLayer::OnUpdate(float deltaTime)
{
    m_current->OnUpdate();
}

void LearnOpenGLLayer::OnImGuiRender()
{
    m_current->OnImgui();
    ImGui::Begin("LearnOpenGLLayer");
    static int e = 0;
    bool bChanged = ImGui::RadioButton("PBR##LearnOpenGL", &e, 0);
    bChanged |= ImGui::RadioButton("BLOOM##LearnOpenGL", &e, 1);
    if(bChanged)
    {
        switch(e)
        {
            case 0: m_current = std::make_shared<PBR>();    break;
            case 1: m_current = std::make_shared<Bloom>();  break;
        }
    }

//     if(ImGui::CollapsingHeader("Light"))
//     {
//         ImGui::Indent();
//         ImGui::SetNextWindowPos({200, 0});
//         if(ImGui::CollapsingHeader("DirectionalLight"))
//         {
//             bool bChanged = ImGui::ColorPicker3("Color", glm::value_ptr(m_dLight.clr));
//             bChanged |= ImGui::DragFloat3("Direction", glm::value_ptr(m_dLight.dir), 0.1f, -1, 1);
//             bChanged |= ImGui::DragFloat("Intensity0", &m_dLight.intensity,  0.1,  0,  10);
//             if(bChanged)
//             {
//                 Renderer::Resources::Get<UniformBuffer>("Light")->Upload("DirectionalLight", &m_dLight);
//             }
//         }
//         if(ImGui::CollapsingHeader("PointLight"))
//         {
//             bool bChanged = ImGui::ColorPicker3("Color", glm::value_ptr(m_pLight.clr));
//             bChanged |= ImGui::DragFloat3("Position", glm::value_ptr(m_pLight.pos),  0.1f,  -10.0,  10.0);
//             bChanged |= ImGui::InputFloat3("AttenuationCoefficents", glm::value_ptr(m_pLight.coe));
//             bChanged |= ImGui::DragFloat("Intensity1", &m_pLight.intensity,  0.1,  0,  10);
//             if(bChanged)
//             {
//                 Renderer::Resources::Get<UniformBuffer>("Light")->Upload("PointLight", &m_pLight);
//             }
//         }
//         if(ImGui::CollapsingHeader("SpotLight"))
//         {
//             bool bChanged = ImGui::ColorPicker3("Color", glm::value_ptr(m_sLight.clr));
//             bChanged |= ImGui::DragFloat3("Position", glm::value_ptr(m_sLight.pos), 0.1f, -10, 10);
//             bChanged |= ImGui::DragFloat3("Direction", glm::value_ptr(m_sLight.dir), 0.1f, -1, 1);
//             bChanged |= ImGui::InputFloat3("AttenuationCoefficents", glm::value_ptr(m_sLight.coe));
//             bChanged |= ImGui::DragFloat("Intensity2", &m_sLight.intensity,  0.1,  0,  10);
// 
//             if(ImGui::DragFloat("InnerCone", &m_sLight.degInnerCone, 1, 0, 60))
//             {
//                 bChanged = true;
//                 m_sLight.cosInnerCone = std::cos(glm::radians(m_sLight.degInnerCone));
//                 if(m_sLight.degOuterCone<m_sLight.degInnerCone)
//                 {
//                     m_sLight.degOuterCone = m_sLight.degInnerCone;
//                     m_sLight.cosOuterCone = m_sLight.cosInnerCone;
//                 }
//             }
//             if(ImGui::DragFloat("OuterCone", &m_sLight.degOuterCone, 1, 0, 90))
//             {
//                 bChanged = true;
//                 m_sLight.cosOuterCone = std::cos(glm::radians(m_sLight.degOuterCone));
//                 if(m_sLight.degInnerCone>m_sLight.degOuterCone)
//                 {
//                     m_sLight.degInnerCone = m_sLight.degOuterCone;
//                     m_sLight.cosInnerCone = m_sLight.cosOuterCone;
//                 }
//             }
// 
//             if(bChanged)
//             {
//                 Renderer::Resources::Get<UniformBuffer>("Light")->Upload("SpotLight", &m_sLight);
//             }
//         }
//         if(ImGui::CollapsingHeader("FlashLight"))
//         {
//             bool bChanged = ImGui::ColorPicker3("Color", glm::value_ptr(m_fLight.clr));
//             bChanged |= ImGui::InputFloat3("AttenuationCoefficents", glm::value_ptr(m_fLight.coe));
//             bChanged |= ImGui::DragFloat("Intensity3", &m_fLight.intensity,  0.1,  0,  10);
//             ImGui::LabelText("Position", "%.1f, %.1f, %.1f", m_fLight.pos.x, m_fLight.pos.y, m_fLight.pos.z);
//             ImGui::LabelText("Direction", "%.1f, %.1f, %.1f", m_fLight.dir.x, m_fLight.dir.y, m_fLight.dir.z);
//             if(ImGui::DragFloat("InnerCone", &m_fLight.degInnerCone, 1, 0, 30))
//             {
//                 bChanged = true;
//                 m_fLight.cosInnerCone = std::cos(glm::radians(m_fLight.degInnerCone));
//                 if(m_fLight.degOuterCone<m_fLight.degInnerCone)
//                 {
//                     m_fLight.degOuterCone = m_fLight.degInnerCone;
//                     m_fLight.cosOuterCone = m_fLight.cosInnerCone;
//                 }
//             }
//             if(ImGui::DragFloat("OuterCone", &m_fLight.degOuterCone, 1, 0, 60))
//             {
//                 bChanged = true;
//                 m_fLight.cosOuterCone = std::cos(glm::radians(m_fLight.degOuterCone));
//                 if(m_fLight.degInnerCone>m_fLight.degOuterCone)
//                 {
//                     m_fLight.degInnerCone = m_fLight.degOuterCone;
//                     m_fLight.cosInnerCone = m_fLight.cosOuterCone;
//                 }
//             }
//             if(bChanged)
//             {
//                 Renderer::Resources::Get<UniformBuffer>("Light")->Upload("FlashLight", &m_fLight);
//             }
//         }
//         ImGui::Unindent();
//     }

    ImGui::End();
}
void LearnOpenGLLayer::_PrepareModel()
{
    m_planet = Renderer::Resources::Create<Model>("Planet")->LoadFromFile("/home/garra/study/dnn/assets/mesh/Planet/planet.obj");
    m_rock = Renderer::Resources::Create<Model>("Rock")->LoadFromFile("/home/garra/study/dnn/assets/mesh/Rock/rock.obj");
//     m_crysisNanoSuit = Renderer::Resources::Create<Model>("CysisNanoSuit")->LoadFromFile("/home/garra/study/dnn/assets/mesh/CysisNanoSuit/scene.fbx");
//     m_silkingMachine = Renderer::Resources::Create<Model>("SilkingMachine")->LoadFromFile("/home/garra/study/dnn/assets/mesh/SilkingMachine/SilkingMachine.fbx");
//     m_horse = Renderer::Resources::Create<Model>("Horse")->LoadFromFile("/home/garra/study/dnn/assets/mesh/Horse/Horse.fbx");
//     m_trailer = Renderer::Resources::Create<Model>("Trailer")->LoadFromFile("/home/garra/study/dnn/assets/mesh/Trailer/Alena_Shek.obj");
//     m_bulb = Renderer::Resources::Create<Model>("Bulb")->LoadFromFile("/home/garra/study/dnn/assets/mesh/Bulb/Bulbs.3ds");
//     m_handLight = Renderer::Resources::Create<Model>("HandLight")->LoadFromFile("/home/garra/study/dnn/assets/mesh/HandLight/hand_light.blend");
//     m_shaderPos = Renderer::Resources::Create<Shader>("Pos")->LoadFromFile("/home/garra/study/dnn/assets/shader/Model.glsl");
//     Renderer::Resources::Create<Shader>("Default")->LoadFromFile("/home/garra/study/dnn/assets/shader/Default.glsl");
//     STOP
}




void LearnOpenGLLayer::_PrepareUniformBuffers()
{
    std::shared_ptr<UniformBuffer> ubTransform = Renderer::Resources::Create<UniformBuffer>("ub_Transform")->SetSize(64);
    ubTransform->Push("WS2CS", glm::ivec2(0, 64));
//     ubTransform->Upload("WS2CS", glm::value_ptr(m_vpBase->GetCamera()->World2Clip()));

    std::shared_ptr<UniformBuffer> ubLight = Renderer::Resources::Create<UniformBuffer>("Light")->SetSize(240);
    ubLight->Push("DirectionalLight", glm::ivec2(0, 32));
    ubLight->Push("PointLight", glm::ivec2(32, 48));
    ubLight->Push("SpotLight", glm::ivec2(80, 64));
    ubLight->Push("FlashLight", glm::ivec2(144, 64));
    ubLight->Upload("DirectionalLight", &m_dLight);
    ubLight->Upload("PointLight", &m_pLight);
    ubLight->Upload("SpotLight", &m_sLight);
    ubLight->Upload("FlashLight", &m_fLight);

    std::shared_ptr<UniformBuffer> ubLights = Renderer::Resources::Create<UniformBuffer>("ub_Lights")->SetSize(64*4);
    ubLights->Push("PointLights", glm::ivec2(0, 64*4));
    struct PointLight
    {
        glm::vec3 pos;
        float padding0;
        glm::vec3 clr;
        float padding1;
    }
    lights[4];

    lights[0].pos = glm::vec3(-10, -10, 10);
    lights[0].clr = glm::vec3(300);
    lights[1].pos = glm::vec3(+10, -10, 10);
    lights[1].clr = glm::vec3(300);
    lights[2].pos = glm::vec3(+10, +10, 10);
    lights[2].clr = glm::vec3(300);
    lights[3].pos = glm::vec3(-10, +10, 10);
    lights[3].clr = glm::vec3(300);
    ubLights->Upload("PointLights", lights);
}



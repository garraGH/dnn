/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : hud.cpp
* author      : Garra
* time        : 2019-10-17 21:19:16
* description : 
*
============================================*/


#include "hud.h"
#include "glm/gtc/type_ptr.hpp"

std::shared_ptr<HUD> HUD::Create()
{
    return std::make_shared<HUD>();
}

HUD::HUD()
{
    m_type = Type::HUD;
    _PrepareResources();
}

void HUD::_PrepareResources()
{
    using MA = Material::Attribute;
    std::shared_ptr<MA> maRadRotated = Renderer::Resources::Create<MA>("RadRotated")->SetType(MA::Type::Float1);
    std::shared_ptr<MA> maRadarRange = Renderer::Resources::Create<MA>("RadarRange")->SetType(MA::Type::Float1);
    std::shared_ptr<MA> maRadarColor = Renderer::Resources::Create<MA>("RadarColor")->Set(MA::Type::Float4, 1, glm::value_ptr(glm::vec4(0, 1, 1, 1.0)));
    std::shared_ptr<MA> maCircleColor = Renderer::Resources::Create<MA>("CircleColor")->Set(MA::Type::Float4, 1, glm::value_ptr(glm::vec4(1.0, 0, 0, 1)));
    std::shared_ptr<MA> maCircleCenter = Renderer::Resources::Create<MA>("CircleCenter")->Set(MA::Type::Float2, 1, glm::value_ptr(glm::vec2(0.1)));
    std::shared_ptr<MA> maCircleRadius = Renderer::Resources::Create<MA>("CircleRadius")->Set(MA::Type::Float1);
    std::shared_ptr<MA> maTime = Renderer::Resources::Create<MA>("Time")->Set(MA::Type::Float1);

    std::shared_ptr<MA> maResolution = Renderer::Resources::Create<MA>("Resolution")->Set(MA::Type::Float2, 1, glm::value_ptr(glm::vec2(1000, 1000)));
    std::shared_ptr<Material> mtr = Renderer::Resources::Create<Material>("HUD");
    mtr->Set("u_Resolution", maResolution);
    mtr->Set("u_RadRotated", maRadRotated);
    mtr->Set("u_RadarRange", maRadarRange);
    mtr->Set("u_RadarColor", maRadarColor);
    mtr->Set("u_CircleColor", maCircleColor);
    mtr->Set("u_CircleCenter", maCircleCenter);
    mtr->Set("u_CircleRadius", maCircleRadius);
    mtr->Set("u_Time", maTime);

    m_radarRange = (float*)maRadarRange->GetData();
    *m_radarRange = 1.0;

    m_time = (float*)maTime->GetData();

    m_circleColor = (float*)maCircleColor->GetData();
    m_circleCenter = (float*)maCircleCenter->GetData();
    m_circleRadius = (float*)maCircleRadius->GetData();
    *m_circleRadius = 0.1;

    Renderer::Resources::Create<Shader>("HUD")->LoadFromFile("/home/garra/study/dnn/assets/shader/HUD.glsl");
}

std::shared_ptr<Material> HUD::GetMaterial() const
{
    return Renderer::Resources::Get<Material>("HUD");
}

std::shared_ptr<Shader> HUD::GetShader() const
{
    return Renderer::Resources::Get<Shader>("HUD");
}

#define MOD(a, b) a-int(a/b)*b
void HUD::OnUpdate(float deltaTime)
{
    m_radRotated += m_speed*deltaTime;
    *m_time = m_timer->GetElapsedTime();
    m_circleCenter[0] = 0.1*sin(*m_time*0.1)+0.1;
    m_circleCenter[1] = 0.1*cos(*m_time*0.1)+0.1;
    *m_circleRadius = MOD(*m_time*0.02, 0.05)+0.008;
    Renderer::Resources::Get<Material::Attribute>("RadRotated")->UpdateData(&m_radRotated);
    Renderer::Resources::Get<Material::Attribute>("CircleCenter")->UpdateData(m_circleCenter);
    Renderer::Resources::Get<Material::Attribute>("CircleRadius")->UpdateData(m_circleRadius);
}

void HUD::OnEvent(Event& e)
{

}

void HUD::OnImGuiRender()
{
    ImGui::PushItemWidth(200);
    ImGui::Separator();
    ImGui::SliderFloat("Speed", &m_speed, 0, 10);
    ImGui::SameLine();
    ImGui::SliderFloat("Range", m_radarRange, 0, 3);
    ImGui::SameLine();
    ImGui::Text("m_radRotated: %.3f",  m_radRotated);
    ImGui::Separator();
    ImGui::ColorPicker4("RadarColor", (float*)Renderer::Resources::Get<Material::Attribute>("RadarColor")->GetData());
    ImGui::SameLine();
    ImGui::ColorPicker4("CircleColor", m_circleColor);


}



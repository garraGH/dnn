/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : pattern.cpp
* author      : Garra
* time        : 2019-10-19 12:10:29
* description : 
*
============================================*/


#include "pattern.h"
#include "glm/gtc/type_ptr.hpp"

std::shared_ptr<Pattern> Pattern::Create()
{
    return std::make_shared<Pattern>();
}

Pattern::Pattern()
{
    m_type = Type::Pattern;
    _PrepareResources();
}

void Pattern::_PrepareResources()
{
    using MU = Material::Uniform;
    std::shared_ptr<MU> maColor = Renderer::Resources::Create<MU>("Color")->Set(MU::Type::Float4);
    std::shared_ptr<MU> maTiles = Renderer::Resources::Create<MU>("Tiles")->Set(MU::Type::Float2);
    std::shared_ptr<MU> maTime = Renderer::Resources::Create<MU>("Time")->Set(MU::Type::Float1);
    std::shared_ptr<Material> mtr = Renderer::Resources::Create<Material>("Pattern");
    mtr->SetUniform("u_Color", maColor);
    mtr->SetUniform("u_Tiles", maTiles);
    mtr->SetUniform("u_Time", maTime);

    m_time = (float*)maTime->GetData();
    m_color = (float*)maColor->GetData();
    m_tiles = (float*)maTiles->GetData();

    *m_time = 0;

    maColor->UpdateData(glm::value_ptr(glm::vec4(1, 0, 0, 1)));
    maTiles->UpdateData(glm::value_ptr(glm::vec2(5, 5)));

    Renderer::Resources::Create<Shader>("Pattern")->LoadFromFile("/home/garra/study/dnn/assets/shader/Pattern.glsl");
}


void Pattern::OnUpdate(float deltaTime)
{
    *m_time += deltaTime;
}

void Pattern::OnEvent(Event& e)
{

}

void Pattern::OnImGuiRender()
{
    ImGui::ColorPicker4("color", m_color);
    ImGui::SliderFloat2("tiles", m_tiles, 0.1, 100);
}


std::shared_ptr<Material> Pattern::GetMaterial() const
{
    return Renderer::Resources::Get<Material>("Pattern");
}

std::shared_ptr<Shader> Pattern::GetShader() const
{
    return Renderer::Resources::Get<Shader>("Pattern");
}




/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : flatcolor.cpp
* author      : Garra
* time        : 2019-10-13 09:50:21
* description : 
*
============================================*/


#include "flatcolor.h"
#include "glm/gtc/type_ptr.hpp"

std::shared_ptr<FlatColor> FlatColor::Create()
{
    return std::make_shared<FlatColor>();
}

FlatColor::FlatColor()
{
    m_type = Type::FlatColor;
    _PrepareResources();
}


void FlatColor::_PrepareResources()
{
    using MU = Material::Uniform;
    std::shared_ptr<MU> maColor = Renderer::Resources::Create<MU>("Color")->Set(MU::Type::Float4, 1, glm::value_ptr(glm::vec4(1, 0, 0, 0)));
    Renderer::Resources::Create<Shader>("FlatColor")->LoadFromFile("/home/garra/study/dnn/assets/shader/FlatColor.glsl");
    Renderer::Resources::Create<Material>("FlatColor")->SetUniform("u_Color", maColor);
}


std::shared_ptr<Material> FlatColor::GetMaterial() const
{
    return Renderer::Resources::Get<Material>("FlatColor");
}

std::shared_ptr<Shader> FlatColor::GetShader() const
{
    return Renderer::Resources::Get<Shader>("FlatColor");
}

void FlatColor::OnImGuiRender()
{
    ImGui::ColorPicker4("FlatColor", (float*)Renderer::Resources::Get<Material::Uniform>("Color")->GetData());
}


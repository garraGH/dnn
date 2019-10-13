/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : colors.cpp
* author      : Garra
* time        : 2019-10-13 09:50:21
* description : 
*
============================================*/


#include "colors.h"
#include "glm/gtc/type_ptr.hpp"

std::shared_ptr<Colors> Colors::Create()
{
    return std::make_shared<Colors>();
}

Colors::Colors()
{
    m_type = Type::Colors;
    _PrepareResources();
}


void Colors::_PrepareResources()
{
    using MA = Material::Attribute;
    std::shared_ptr<MA> maColorFst = Renderer::Resources::Create<MA>("ColorFst")->Set(MA::Type::Float4, glm::value_ptr(glm::vec4(0.8, 0.1, 0.2, 1.0)));
    std::shared_ptr<MA> maColorSnd = Renderer::Resources::Create<MA>("ColorSnd")->Set(MA::Type::Float4, glm::value_ptr(glm::vec4(0.1, 0.8, 0.8, 1.0)));
    std::shared_ptr<MA> maTime = Renderer::Resources::Create<MA>("Time")->SetType(MA::Type::Float1);
    std::shared_ptr<MA> maSpeed = Renderer::Resources::Create<MA>("Speed")->SetType(MA::Type::Float1);
    std::shared_ptr<MA> maEasingFunction = Renderer::Resources::Create<MA>("EasingFunction")->SetType(MA::Type::Int1);
    std::shared_ptr<Material> mtr = Renderer::Resources::Create<Material>("Colors");
    mtr->Set("u_ColorFst", maColorFst);
    mtr->Set("u_ColorSnd", maColorSnd);
    mtr->Set("u_Time", maTime);
    mtr->Set("u_Speed", maSpeed);
    mtr->Set("u_EasingFunction", maEasingFunction);

    m_speed = (float*)maSpeed->GetData();
    m_easingFunction = (EasingFunction*)maEasingFunction->GetData();
    *m_speed = 1.0;
    *m_easingFunction = EasingFunction::Linear;

    Renderer::Resources::Create<Shader>("Colors")->LoadFromFile("/home/garra/study/dnn/assets/shader/Colors.glsl");
}


std::shared_ptr<Material> Colors::GetMaterial() const
{
    return Renderer::Resources::Get<Material>("Colors");
}

std::shared_ptr<Shader> Colors::GetShader() const
{
    return Renderer::Resources::Get<Shader>("Colors");
}

void Colors::OnUpdate()
{
    float t = m_timer->GetElapsedTime();
    Renderer::Resources::Get<Material::Attribute>("Time")->UpdateData((void*)&t);
}


void Colors::OnImGuiRender()
{
    ImGui::PushItemWidth(200);
    ImGui::ColorPicker4("ColorFst", (float*)Renderer::Resources::Get<Material::Attribute>("ColorFst")->GetData());
    ImGui::SameLine();
    ImGui::ColorPicker4("ColorSnd", (float*)Renderer::Resources::Get<Material::Attribute>("ColorSnd")->GetData());
    ImGui::SameLine();
    ImGui::SliderFloat("Speed", m_speed, 0.1, 10);

#define RadioButton(x) \
    if(ImGui::RadioButton(#x, *m_easingFunction == EasingFunction::x))\
    {                                                                 \
        *m_easingFunction = EasingFunction::x;                        \
    }                                                                 \

    RadioButton(Linear)
    RadioButton(ExponentialIn)
    RadioButton(ExponentialOut)
    RadioButton(ExponentialInOut)
    RadioButton(SineIn)
    RadioButton(SineOut)
    RadioButton(SineInOut)
    RadioButton(QinticIn)
    RadioButton(QinticOut)
    RadioButton(QinticInOut)
    RadioButton(QuarticIn)
    RadioButton(QuarticOut)
    RadioButton(QuarticInOut)
    RadioButton(QuadraticIn)
    RadioButton(QuadraticOut)
    RadioButton(QuadraticInOut)
    RadioButton(CubicIn)
    RadioButton(CubicOut)
    RadioButton(CubicInOut)
    RadioButton(ElasticIn)
    RadioButton(ElasticOut)
    RadioButton(ElasticInOut)
    RadioButton(CircularIn)
    RadioButton(CircularOut)
    RadioButton(CircularInOut)
    RadioButton(BounceIn)
    RadioButton(BounceOut)
    RadioButton(BounceInOut)
    RadioButton(BackIn)
    RadioButton(BackOut)
    RadioButton(BackInOut)
}



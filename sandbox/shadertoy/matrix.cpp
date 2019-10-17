/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : matrix.cpp
* author      : Garra
* time        : 2019-10-17 16:11:05
* description : 
*
============================================*/


#include "matrix.h"
#include "glm/gtc/type_ptr.hpp"

std::shared_ptr<Matrix> Matrix::Create()
{
    return std::make_shared<Matrix>();
}

Matrix::Matrix()
{
    m_type = Type::Matrix;
    _PrepareResources();
}

void Matrix::_PrepareResources()
{
    using MA = Material::Attribute;
    std::shared_ptr<MA> maResolution = Renderer::Resources::Create<MA>("Resolution")->Set(MA::Type::Float2, 1, glm::value_ptr(glm::vec2(1000, 1000)));
    std::shared_ptr<MA> maRadius = Renderer::Resources::Create<MA>("Radius")->Set(MA::Type::Float1);
    std::shared_ptr<MA> maTime = Renderer::Resources::Create<MA>("Time")->Set(MA::Type::Float1);
    std::shared_ptr<MA> maStyle = Renderer::Resources::Create<MA>("Style")->Set(MA::Type::Int1);
    std::shared_ptr<MA> maSpeed = Renderer::Resources::Create<MA>("Speed")->Set(MA::Type::Float1);
    std::shared_ptr<MA> maShowSDF = Renderer::Resources::Create<MA>("ShowSDF")->Set(MA::Type::Int1, 1, &m_showSDF);

    m_radius = (float*)maRadius->GetData();
    *m_radius = 0.35;

    m_style = (Style*)maStyle->GetData();
    *m_style = Style::T;

    m_speed = (float*)maSpeed->GetData();
    *m_speed = 0.1;

    std::shared_ptr<Material> mtrMatrix = Renderer::Resources::Create<Material>("Matrix");
    mtrMatrix->Set("u_Resolution", maResolution);
    mtrMatrix->Set("u_Radius", maRadius);
    mtrMatrix->Set("u_Time", maTime);
    mtrMatrix->Set("u_Style", maStyle);
    mtrMatrix->Set("u_Speed", maSpeed);
    mtrMatrix->Set("u_ShowSDF", maShowSDF);


    Renderer::Resources::Create<Shader>("Matrix")->LoadFromFile("/home/garra/study/dnn/assets/shader/Matrix.glsl");
}

std::shared_ptr<Material> Matrix::GetMaterial() const
{
    return Renderer::Resources::Get<Material>("Matrix");
}

std::shared_ptr<Shader> Matrix::GetShader() const
{
    return Renderer::Resources::Get<Shader>("Matrix");
}

void Matrix::OnUpdate()
{
    float t = m_timer->GetElapsedTime();
    Renderer::Resources::Get<Material::Attribute>("Time")->UpdateData(&t);
}

void Matrix::OnImGuiRender()
{
#define RadioButton(x) \
    if(ImGui::RadioButton(#x, *m_style == Style::x)) \
    {                                                \
        *m_style = Style::x;                         \
    }                                                \

    RadioButton(R);
    ImGui::SameLine();
    RadioButton(T);
    ImGui::SameLine();
    RadioButton(S);
    ImGui::SameLine();
    RadioButton(TR);
    ImGui::SameLine();
    RadioButton(TS);
    ImGui::SameLine();
    RadioButton(RS);
    ImGui::SameLine();
    RadioButton(TRS);

#undef RadioButton
    ImGui::Separator();
    if(ImGui::RadioButton("ShowSDF", m_showSDF))
    {
        m_showSDF = !m_showSDF;
        Renderer::Resources::Get<Material::Attribute>("ShowSDF")->UpdateData(&m_showSDF);
    }
    ImGui::Separator();
    ImGui::PushItemWidth(200);
    ImGui::SliderFloat("radius", m_radius, 0.05, 0.5);
    ImGui::SameLine();
    ImGui::SliderFloat("speed", m_speed, 0, 10);
}

void Matrix::OnEvent(Event& e)
{
    EventDispatcher ed(e);
#define DISPATCH(event) ed.Dispatch<event>(BIND_EVENT_CALLBACK(Matrix, _On##event))
    DISPATCH(MouseButtonPressedEvent);
    DISPATCH(MouseButtonReleasedEvent);
    DISPATCH(MouseMovedEvent);
    DISPATCH(MouseScrolledEvent);
#undef DISPATCH
}

bool Matrix::_OnMouseButtonPressedEvent(MouseButtonPressedEvent& e)
{
    return false;
}


bool Matrix::_OnMouseButtonReleasedEvent(MouseButtonReleasedEvent& e)
{
    return false;
}

bool Matrix::_OnMouseMovedEvent(MouseMovedEvent& e)
{
    return false;
}

bool Matrix::_OnMouseScrolledEvent(MouseScrolledEvent& e)
{
    return false;
}

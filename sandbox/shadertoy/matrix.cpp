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
    using MU = Material::Uniform;
    std::shared_ptr<MU> maResolution = Renderer::Resources::Create<MU>("Resolution")->Set(MU::Type::Float2, 1, glm::value_ptr(glm::vec2(1000, 1000)));
    std::shared_ptr<MU> maRadius = Renderer::Resources::Create<MU>("Radius")->Set(MU::Type::Float1);
    std::shared_ptr<MU> maTime = Renderer::Resources::Create<MU>("Time")->Set(MU::Type::Float1);
    std::shared_ptr<MU> maStyle = Renderer::Resources::Create<MU>("Style")->Set(MU::Type::Int1);
    std::shared_ptr<MU> maSpeed = Renderer::Resources::Create<MU>("Speed")->Set(MU::Type::Float1);
    std::shared_ptr<MU> maShowSDF = Renderer::Resources::Create<MU>("ShowSDF")->Set(MU::Type::Int1, 1, &m_showSDF);

    m_radius = (float*)maRadius->GetData();
    *m_radius = 0.35;

    m_style = (Style*)maStyle->GetData();
    *m_style = Style::T;

    m_speed = (float*)maSpeed->GetData();
    *m_speed = 0.1;

    std::shared_ptr<Material> mtrMatrix = Renderer::Resources::Create<Material>("Matrix");
    mtrMatrix->SetUniform("u_Resolution", maResolution);
    mtrMatrix->SetUniform("u_Radius", maRadius);
    mtrMatrix->SetUniform("u_Time", maTime);
    mtrMatrix->SetUniform("u_Style", maStyle);
    mtrMatrix->SetUniform("u_Speed", maSpeed);
    mtrMatrix->SetUniform("u_ShowSDF", maShowSDF);


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

void Matrix::OnUpdate(float deltaTime)
{
    float t = m_timer->GetElapsedTime();
    Renderer::Resources::Get<Material::Uniform>("Time")->UpdateData(&t);
}

void Matrix::OnImGuiRender()
{
#define RadioButton(x) \
    if(ImGui::RadioButton(#x, *m_style == Style::x)) \
    {                                                \
        *m_style = Style::x;                         \
    }                                                \

    RadioButton(T);
    ImGui::SameLine();
    RadioButton(R);
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
        Renderer::Resources::Get<Material::Uniform>("ShowSDF")->UpdateData(&m_showSDF);
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

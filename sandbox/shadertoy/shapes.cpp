/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : shapes.cpp
* author      : Garra
* time        : 2019-10-13 09:50:21
* description : 
*
============================================*/


#include "shapes.h"
#include "glm/gtc/type_ptr.hpp"

std::shared_ptr<Shapes> Shapes::Create()
{
    return std::make_shared<Shapes>();
}

Shapes::Shapes()
{
    m_type = Type::Shapes;
    _PrepareResources();
}

void Shapes::_PrepareResources()
{
    using MA = Material::Attribute;
    std::shared_ptr<MA> maResolution = Renderer::Resources::Create<MA>("Resolution")->Set(MA::Type::Float2, 1, glm::value_ptr(glm::vec2(1000, 1000)));
    std::shared_ptr<MA> maStyle = Renderer::Resources::Create<MA>("Style")->Set(MA::Type::Int1);
    std::shared_ptr<MA> maOperator = Renderer::Resources::Create<MA>("Operator")->Set(MA::Type::Int1);
    std::shared_ptr<MA> maMode = Renderer::Resources::Create<MA>("Mode")->Set(MA::Type::Int1);
    std::shared_ptr<MA> maLineWidth = Renderer::Resources::Create<MA>("LineWidth")->Set(MA::Type::Float1);
    std::shared_ptr<MA> maTorusWidth = Renderer::Resources::Create<MA>("TorusWidth")->Set(MA::Type::Float1);
    std::shared_ptr<MA> maNumber = Renderer::Resources::Create<MA>("Number")->Set(MA::Type::Float1);
    std::shared_ptr<MA> maRoundRadius = Renderer::Resources::Create<MA>("RoundRadius")->Set(MA::Type::Float1);
    std::shared_ptr<MA> maSawTooth = Renderer::Resources::Create<MA>("SawTooth")->Set(MA::Type::Float1);


    std::shared_ptr<MA> maPoints = Renderer::Resources::Create<MA>("Points")->Set(MA::Type::Float2, 2);
    std::shared_ptr<MA> maColors = Renderer::Resources::Create<MA>("Colors")->Set(MA::Type::Float4, 2);

    m_style = (Style*)maStyle->GetData();
    *m_style = Style::Box;

    m_operator = (Operator*)maOperator->GetData();
    *m_operator = Operator::Union;

    m_mode = (Mode*)maMode->GetData();
    *m_mode = Mode::Fill;

    m_lineWidth = (float*)maLineWidth->GetData();
    *m_lineWidth = 2;

    m_points = (Points*)maPoints->GetData();
    m_points->pnt1 = glm::vec2(0.4);
    m_points->pnt2 = glm::vec2(0.6);
    
    m_colors = (Colors*)maColors->GetData();
    m_colors->cFilled = glm::vec4(0.2, 0.3, 0.5, 1.0);
    m_colors->cOutline = glm::vec4(0.5, 0.3, 0.2, 1.0);

    m_torusWidth = (float*)maTorusWidth->GetData();
    *m_torusWidth = 0.1;

    m_number = (float*)maNumber->GetData();
    *m_number = 3.0;

    m_roundRadius = (float*)maRoundRadius->GetData();
    *m_roundRadius = 0.1;

    m_sawTooth = (float*)maSawTooth->GetData();
    *m_sawTooth = 0.5;

    Renderer::Resources::Create<Shader>("Shapes")->LoadFromFile("/home/garra/study/dnn/assets/shader/Shapes.glsl");
    std::shared_ptr<Material> mtrShapes = Renderer::Resources::Create<Material>("Shapes");
    mtrShapes->Set("u_Resolution", maResolution);
    mtrShapes->Set("u_Style", maStyle);
    mtrShapes->Set("u_Operator", maOperator);
    mtrShapes->Set("u_Mode", maMode);
    mtrShapes->Set("u_Points", maPoints);
    mtrShapes->Set("u_Colors", maColors);
    mtrShapes->Set("u_LineWidth", maLineWidth);
    mtrShapes->Set("u_TorusWidth", maTorusWidth);
    mtrShapes->Set("u_Number", maNumber);
    mtrShapes->Set("u_RoundRadius", maRoundRadius);
    mtrShapes->Set("u_SawTooth", maSawTooth);
}

std::shared_ptr<Material> Shapes::GetMaterial() const
{
    return Renderer::Resources::Get<Material>("Shapes");
}

std::shared_ptr<Shader> Shapes::GetShader() const
{
    return Renderer::Resources::Get<Shader>("Shapes");
}

void Shapes::OnImGuiRender()
{
#define RadioButton(variable, E, value)                         \
    if(ImGui::RadioButton(#value, *m_##variable == E::value))   \
    {                                                           \
        *m_##variable = E::value;                               \
    }                                                           \

    RadioButton(style, Style, Line)
    ImGui::SameLine();
    RadioButton(style, Style, Segment)
    ImGui::SameLine();
    RadioButton(style, Style, Box)
    ImGui::SameLine();
    RadioButton(style, Style, RoundBox)
    ImGui::SameLine();
    RadioButton(style, Style, Circle)
    ImGui::SameLine();
    RadioButton(style, Style, Elipse)
    ImGui::SameLine();
    RadioButton(style, Style, Torus)
    ImGui::SameLine();
    RadioButton(style, Style, Polygon)
    ImGui::SameLine();
    RadioButton(style, Style, Petal)
    ImGui::SameLine();
    RadioButton(style, Style, Topology)
        
    ImGui::Separator();

    RadioButton(mode, Mode, Fill)
    ImGui::SameLine();
    RadioButton(mode, Mode, Outline)
    ImGui::SameLine();
    RadioButton(mode, Mode, Both)
    ImGui::Separator();


    ImGui::PushItemWidth(200);
    ImGui::ColorPicker4("FillColor", (float*)&m_colors->cFilled);
    ImGui::SameLine();
    ImGui::ColorPicker4("OutlineColor", (float*)&m_colors->cOutline);
    ImGui::Separator();
    ImGui::SliderFloat("LineWidth", m_lineWidth, 1, 32);
    ImGui::DragFloat2("Point1", (float*)&m_points->pnt1, 0.01, 0, 1);
    ImGui::DragFloat2("Point2", (float*)&m_points->pnt2, 0.01, 0, 1);

    
    if(*m_style == Style::RoundBox)
    {
        ImGui::DragFloat("RoundRadius", m_roundRadius,  0.01,  0,  1);
    }
    if(*m_style == Style::Torus)
    {
        ImGui::DragFloat("Width", m_torusWidth,  0.01,  0,  1);
    }
    if(*m_style == Style::Polygon || *m_style == Style::Petal)
    {
        ImGui::DragFloat("Number", m_number,  0.05, 1, 20);
        ImGui::DragFloat("SawTooth", m_sawTooth,  0.01, 0, 1);
    }

    ImGui::Separator();
    if(*m_style == Style::Topology)
    {
        RadioButton(operator, Operator, Union);
        ImGui::SameLine();
        RadioButton(operator, Operator, Intersect);
        ImGui::SameLine();
        RadioButton(operator, Operator, Diffrence);
    }
    ImGui::Separator();
#undef RadioButton
}

void Shapes::OnEvent(Event& e)
{
    EventDispatcher ed(e);
#define DISPATCH(event) ed.Dispatch<event>(BIND_EVENT_CALLBACK(Shapes, _On##event))
    DISPATCH(MouseButtonPressedEvent);
    DISPATCH(MouseButtonReleasedEvent);
    DISPATCH(MouseMovedEvent);
    DISPATCH(MouseScrolledEvent);
#undef DISPATCH
}

bool Shapes::_OnMouseButtonPressedEvent(MouseButtonPressedEvent& e)
{
    m_bPressed = true;
    m_points->pnt1.x = Input::GetMouseX()/1000.0;
    m_points->pnt1.y = 1.0-Input::GetMouseY()/1000.0;
    m_points->pnt2 = m_points->pnt1;
    return false;
}


bool Shapes::_OnMouseButtonReleasedEvent(MouseButtonReleasedEvent& e)
{
    m_bPressed = false;
    return false;
}

bool Shapes::_OnMouseMovedEvent(MouseMovedEvent& e)
{
    if(m_bPressed)
    {
        m_points->pnt2.x = e.GetX()/1000.0;
        m_points->pnt2.y = 1.0-e.GetY()/1000.0;
    }
    return false;
}

bool Shapes::_OnMouseScrolledEvent(MouseScrolledEvent& e)
{
    switch(*m_style)
    {
        case Style::Torus: 
            *m_torusWidth += e.GetOffsetY()*0.01;
            break;
        case Style::RoundBox:
            *m_roundRadius += e.GetOffsetY()*0.05;
            break;
        case Style::Polygon:
        case Style::Petal:
            if(Input::IsKeyPressed(KEY_LEFT_CONTROL))
            {
                *m_sawTooth += e.GetOffsetY()*0.01;
            }
            else
            {
                *m_number += e.GetOffsetY()*0.1;
            }
            break;
        default: break;
    }
    return false;
}

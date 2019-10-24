/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : ./renderer/viewport/viewport.cpp
* author      : Garra
* time        : 2019-10-25 11:47:28
* description : 
*
============================================*/


#include "viewport.h"
#include "imgui.h"
#include "../../core.h"

std::shared_ptr<Viewport> Viewport::Create(const std::string& name)
{
    return std::make_shared<Viewport>(name);
}

Viewport::Viewport(const std::string& name) 
    : m_name(name) 
{ 
    m_camera = Camera::Create(m_name+"camera");
}

void Viewport::SetRange(float left, float bottom, float width, float height)
{
    m_range = {left, bottom, width, height};
    m_camera->SetAspectRatio(width/height);
}

void Viewport::SetType(Type t)
{
    if(m_type == t)
    {
        return;
    }
    
    m_type = t;
    if(m_type == Type::Percentage)
    {
        m_range[0] /= m_windowSize[0];
        m_range[1] /= m_windowSize[1];
        m_range[2] /= m_windowSize[0];
        m_range[3] /= m_windowSize[1];
    }
    else
    {
        m_range[0] *= m_windowSize[0];
        m_range[1] *= m_windowSize[1];
        m_range[2] *= m_windowSize[0];
        m_range[3] *= m_windowSize[1];
    }
}

std::array<float, 4> Viewport::GetRange() const
{
    if(m_type == Type::Fixed)
    {
        return m_range;
    }

    return { m_range[0]*m_windowSize[0], m_range[1]*m_windowSize[1], m_range[2]*m_windowSize[0], m_range[3]*m_windowSize[1]};
}

const std::shared_ptr<Camera>& Viewport::GetCamera() const
{
    return m_camera;
}

void Viewport::OnUpdate(float deltaTime)
{
    m_camera->OnUpdate(deltaTime);
}

void Viewport::OnEvent(Event& e)
{
    EventDispatcher ed(e);
#define DISPATCH(event) ed.Dispatch<event>(BIND_EVENT_CALLBACK(Viewport, _On##event))
    DISPATCH(WindowResizeEvent);
#undef DISPATCH
    m_camera->OnEvent(e);
}

bool Viewport::_OnWindowResizeEvent(WindowResizeEvent& e)
{
    m_windowSize[0] = e.GetWidth();
    m_windowSize[1] = e.GetHeight();
    if(m_type == Type::Percentage)
    {
        m_camera->SetAspectRatio((m_windowSize[0]*m_range[2])/(m_windowSize[1]*m_range[3]));
    }
    return false;
}

void Viewport::OnImGuiRender()
{
    ImGui::Begin(m_name.c_str());

    if(ImGui::RadioButton("Fixed", m_type == Type::Fixed))
    {
        SetType(Type::Fixed);
    }
    ImGui::SameLine();
    if(ImGui::RadioButton("Percentage", m_type == Type::Percentage))
    {
        SetType(Type::Percentage);
    }
    ImGui::Separator();
    if(ImGui::InputFloat4("range", &m_range[0], ImGuiInputTextFlags_EnterReturnsTrue))
    {
        m_camera->SetAspectRatio(m_range[2]/m_range[3]);
    }
    ImGui::Separator();

    m_camera->OnImGuiRender();

    ImGui::End();
}

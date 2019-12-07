/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : ./renderer/viewport/viewport.cpp
* author      : Garra
* time        : 2019-10-25 11:47:28
* description : 
*
============================================*/


#include "input/input.h"
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
    m_cameraDefault = Camera::Create(m_name+"-DefaultCamera");
}

void Viewport::AttachCamera(const std::shared_ptr<Camera>& camera)
{
    m_cameraAttached = camera;
}

void Viewport::DetachCamera()
{
    m_cameraAttached = nullptr;
}

std::shared_ptr<Viewport> Viewport::SetRange(float left, float bottom, float width, float height)
{
    if(width>1||height>1)
    {
        m_type = Type::Fixed;
    }

    m_range = {left, bottom, width, height};
    auto r = GetRange();
    m_cameraDefault->SetAspectRatio(r[2]/r[3]);
//     if(m_cameraAttached) 
//     {
//         m_cameraAttached->SetAspectRatio(width/height);
//     }

    return shared_from_this();
}


std::shared_ptr<Viewport> Viewport::SetType(Type t)
{
    if(m_type == t)
    {
        return shared_from_this();
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

    return shared_from_this();
}

std::shared_ptr<Viewport> Viewport::SetBackgroundColor(float r, float g, float b, float a)
{
    m_backgroundColor = {r, g, b, a};
    return shared_from_this();
}

std::shared_ptr<Viewport> Viewport::SetBackgroundDepth(float depth)
{
    m_backgroundDepth = depth;
    return shared_from_this();
}

const std::string& Viewport::GetName() const
{
    return m_name;
}

float Viewport::GetWidth() const
{
    return GetRange()[2];
}

float Viewport::GetHeight() const
{
    return GetRange()[3];
}

std::array<float, 4> Viewport::GetRange() const
{
    if(m_type == Type::Percentage)
    {
        return { m_range[0]*m_windowSize[0], m_range[1]*m_windowSize[1], m_range[2]*m_windowSize[0], m_range[3]*m_windowSize[1]};
    }

    return m_range;
}

const std::array<float, 4>& Viewport::GetBackgroundColor() const
{
    return m_backgroundColor;
}

float Viewport::GetBackgroundDepth() const
{
    return m_backgroundDepth;
}

const std::shared_ptr<Camera>& Viewport::GetCamera() const
{
    return m_cameraAttached? m_cameraAttached : m_cameraDefault;
}

bool Viewport::_CursorOutside() const
{
    auto [x, y] = Input::GetMousePosition();
    y = m_windowSize[1]-y;
    auto [left, bottom, width, height] = GetRange();
    return x<left || x>left+width || y<bottom || y>bottom+height;
}

void Viewport::OnUpdate(float deltaTime)
{
//     if(_CursorOutside())
//     {
//         return;
//     }

    GetCamera()->OnUpdate(deltaTime);
}

void Viewport::OnEvent(Event& e)
{
    if((e.GetType() == EventType::ET_MouseScrolled || e.GetType() == EventType::ET_MouseButtonPressed) && _CursorOutside())
    {
        return;
    }

    EventDispatcher ed(e);
#define DISPATCH(event) ed.Dispatch<event>(BIND_EVENT_CALLBACK(Viewport, _On##event))
    DISPATCH(WindowResizeEvent);
#undef DISPATCH

    GetCamera()->OnEvent(e, this);
}

bool Viewport::_OnWindowResizeEvent(WindowResizeEvent& e)
{
    if(m_type == Type::Constant)
    {
        return false;
    }

    m_windowSize[0] = e.GetWidth();
    m_windowSize[1] = e.GetHeight();
    if(m_type == Type::Percentage)
    {
        float asp = (m_windowSize[0]*m_range[2])/(m_windowSize[1]*m_range[3]);
        m_cameraDefault->SetAspectRatio(asp);
//         if(m_cameraAttached)
//         {
//             m_cameraAttached->SetAspectRatio(asp);
//         }
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
    if(ImGui::InputFloat4("range", &m_range[0], "%.3f", ImGuiInputTextFlags_EnterReturnsTrue))
    {
        auto r = GetRange();
        m_cameraDefault->SetAspectRatio(r[2]/r[3]);
//         if(m_cameraAttached)
//         {
//             m_cameraAttached->SetAspectRatio(asp);
//         }
    }
    ImGui::Separator();

//     if(m_cameraAttached)
//     {
//         m_cameraAttached->OnImGuiRender(false);
//     }
//     else
//     {
        m_cameraDefault->OnImGuiRender(false);
//     }

    ImGui::End();
}



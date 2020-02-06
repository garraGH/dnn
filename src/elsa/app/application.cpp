/*============================================;
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : application.cpp
* author      : Garra
* time        : 2019-09-24 10:44:51
* description : 
*
============================================*/


#include <stdio.h>
#include "glad/gl.h"
#include "application.h"
#include "logger.h"
#include "timer_cpu.h"
#include "core.h"
#include "../input/input.h"
#include "../renderer/renderer.h"
Application* Application::s_instance = nullptr;
std::shared_ptr<Camera> Application::s_camera = nullptr;

Application::Application()
    : m_running(true)
{
    PROFILE_FUNCTION
    CORE_ASSERT(!s_instance, "Application already exist!");
    s_instance = this;
    s_camera = std::make_shared<Camera>("GlobalMainCamera");

    m_window = std::unique_ptr<Window>(Window::Create(WindowsProps("Elsa", 1000, 1000)));
    m_window->SetEventCallback(BIND_EVENT_CALLBACK(Application, OnEvent));

    m_layerStack = LayerStack::Create();
    m_layerImGui = ImGuiLayer::Create();
    PushOverlay(m_layerImGui);

    
#define REGISTER_KEY_PRESSED_FUNCTION(key) m_keyPressed[KEY_##key] = std::bind(&Application::_OnKeyPressed_##key, this, std::placeholders::_1)
#define REGISTER_KEY_RELEASED_FUNCTION(key) m_keyReleased[KEY_##key] = std::bind(&Application::_OnKeyReleased_##key, this)
//     REGISTER_KEY_PRESSED_FUNCTION(a);
//     REGISTER_KEY_PRESSED_FUNCTION(R);
//     REGISTER_KEY_RELEASED_FUNCTION(Q);
    REGISTER_KEY_RELEASED_FUNCTION(q);
    REGISTER_KEY_PRESSED_FUNCTION(ESCAPE);
    REGISTER_KEY_PRESSED_FUNCTION(SPACE);
    REGISTER_KEY_PRESSED_FUNCTION(V);
#undef REGISTER_KEY_PRESSED_FUNCTION
#undef REGISTER_KEY_RELEASED_FUNCTION
}

Application::~Application()
{
    CORE_TRACE("{}", "Application destructed.");
}

void Application::OnEvent(Event& e)
{
    CORE_TRACE("{}", e);
    EventDispatcher ed(e);
#define DISPATCH(event) ed.Dispatch<event>(BIND_EVENT_CALLBACK(Application, _On##event))
    DISPATCH(WindowCloseEvent);
    DISPATCH(KeyPressedEvent);
    DISPATCH(KeyReleasedEvent);
#undef DISPATCH

    // pass application event down, only capture key or mouse event
    if(m_layerImGui->CaptureInput() && !e.IsCategory(EventCategory::EC_Application))
    {
        return;
    }

    for(auto it = m_layerStack->end(); it != m_layerStack->begin();)
    {
        (*--it)->OnEvent(e);
        if(e.IsHandled())
        {
            break;
        }
    }
}

#define _ON(event) bool Application::_On##event(event& e)
_ON(WindowCloseEvent)
{
    INFO("{}", "CLOSED");
    m_running = false;
    return true;
}

_ON(KeyPressedEvent)
{
    std::function<bool(int)> fn = m_keyPressed[e.GetKeyCode()];
    return fn == nullptr? false : fn(e.GetRepeatCount());
}

_ON(KeyReleasedEvent)
{
    std::function<bool()> fn = m_keyReleased[e.GetKeyCode()];
    return fn == nullptr? false : fn();
}

#undef _ON

void Application::Run()
{
    PROFILE_FUNCTION
    INFO("{}", "Application::Run");
    WindowResizeEvent e(1280, 720);
    if(e.IsCategory(EC_Application))
    {
        TRACE("{}", e);
    }
    if(e.IsCategory(EC_Input))
    {
        TRACE("{}", e);
    }

    while(m_running)
    {
        TimerCPU::NeedProfile()? _RunWithProfile() : _RunNeatly();
    }
}

void Application::_RunWithProfile()
{
    PROFILE_INCREASE
    PROFILE_SCOPE("RunLoop")

    for(auto& layer : *m_layerStack.get())
    {
        PROFILE_SCOPE(layer->GetName());
        layer->OnUpdate(m_timer->GetDeltaTime());
    }

    {
        PROFILE_SCOPE("ImGui");
        m_layerImGui->Begin();
        for(auto& layer : *m_layerStack.get())
        {
            layer->OnImGuiRender();
        }
        m_layerImGui->End();

    }
    m_window->OnUpdate();
}

void Application::_RunNeatly()
{
    for(auto& layer : *m_layerStack.get())
    {
        layer->OnUpdate(m_timer->GetDeltaTime());
    }

    m_layerImGui->Begin();
    for(auto& layer : *m_layerStack.get())
    {
        layer->OnImGuiRender();
    }
    m_layerImGui->End();

    m_window->OnUpdate();
}

bool Application::_OnKeyPressed_a(int repeatCount)
{
    INFO("a pressed: {}", repeatCount);
    return true;
}

bool Application::_OnKeyPressed_R(int repeatCount)
{
    INFO("R pressed: {}", repeatCount);
    return true;
}

bool Application::_OnKeyReleased_q()
{
    INFO("{}", "q released");
    m_running = false;
    return true;
}

bool Application::_OnKeyReleased_Q()
{
    INFO("{}", "Q released");
    m_running = false;
    return true;
}

bool Application::_OnKeyPressed_V(int repeatCount)
{
    m_window->SwitchVSync();
    return true;
}

bool Application::_OnKeyPressed_ESCAPE(int repeatCount)
{
    m_running = false;
    return true;
}

bool Application::_OnKeyPressed_SPACE(int repeatCount)
{
    m_window->SwitchFullscreen();
    return true;
}

void Application::PushLayer(const std::shared_ptr<Layer>& layer)
{
    layer->OnAttach();
    m_layerStack->PushLayer(layer);
}

void Application::PushOverlay(const std::shared_ptr<Layer>& layer)
{
    layer->OnAttach();
    m_layerStack->PushOverlay(layer);
}

#undef BIND_EVENT_CALLBACK


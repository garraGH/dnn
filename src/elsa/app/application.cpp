/*============================================
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

#define BIND_EVENT_CALLBACK(x) std::bind(&Application::x,  this, std::placeholders::_1)
Application* Application::s_instance = nullptr;

Application::Application()
    : m_running(true)
{
    CORE_ASSERT(!s_instance, "Application already exist!");
    s_instance = this;

    m_window = std::unique_ptr<Window>(Window::Create());
    m_window->SetEventCallback(BIND_EVENT_CALLBACK(OnEvent));

    m_layerImGui = new ImGuiLayer;
    PushOverlay(m_layerImGui);

#define REGISTER_KEY_PRESSED_FUNCTION(key) m_keyPressed[#key[0]] = std::bind(&Application::_OnKeyPressed_##key, this, std::placeholders::_1)
#define REGISTER_KEY_RELEASED_FUNCTION(key) m_keyReleased[#key[0]] = std::bind(&Application::_OnKeyReleased_##key, this)
    REGISTER_KEY_PRESSED_FUNCTION(a);
    REGISTER_KEY_PRESSED_FUNCTION(R);
    REGISTER_KEY_RELEASED_FUNCTION(q);
    REGISTER_KEY_RELEASED_FUNCTION(Q);
#undef REGISTER_KEY_PRESSED_FUNCTION
#undef REGISTER_KEY_RELEASED_FUNCTION
}

Application::~Application()
{
    CORE_TRACE("Application destructed.");
}

void Application::OnEvent(Event& e)
{
    CORE_TRACE("{0}", e);
    EventDispatcher ed(e);
#define DISPATCH(event) ed.Dispatch<event>(BIND_EVENT_CALLBACK(_On##event))
    DISPATCH(WindowCloseEvent);
    DISPATCH(KeyPressedEvent);
    DISPATCH(KeyReleasedEvent);
#undef DISPATCH
    for(auto it = m_layerStack.end(); it != m_layerStack.begin();)
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
    INFO("CLOSED");
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
    TimerCPU t("Application::Run");
    INFO("Application::Run\n");
    WindowResizeEvent e(1280, 720);
    if(e.IsCategory(EC_Application))
    {
        TRACE(e);
    }
    if(e.IsCategory(EC_Input))
    {
        TRACE(e);
    }

    while(m_running)
    {
        glClearColor(1, 0, 1, 1);
        glClear(GL_COLOR_BUFFER_BIT);
        for(Layer* layer : m_layerStack)
        {
            layer->OnUpdate();
        }

        m_layerImGui->Begin();
        for(Layer* layer : m_layerStack)
        {
            layer->OnImGuiRender();
        }
        m_layerImGui->End();

        m_window->OnUpdate();

        auto[x, y] = Input::GetMousePosition();
        CORE_TRACE("{0}, {0}", x, y);
    }
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
    INFO("q released");
    m_running = false;
    return true;
}

bool Application::_OnKeyReleased_Q()
{
    INFO("Q released");
    m_running = false;
    return true;
}

void Application::PushLayer(Layer* layer)
{
    layer->OnAttach();
    m_layerStack.PushLayer(layer);
}

void Application::PushOverlay(Layer* layer)
{
    layer->OnAttach();
    m_layerStack.PushOverlay(layer);
}

#undef BIND_EVENT_CALLBACK


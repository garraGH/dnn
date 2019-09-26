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
#include "application.h"
#include "glfw3.h"
#include "logger.h"
#include "timer_cpu.h"

#define BIND_EVENT_CALLBACK(x) std::bind(&Application::x,  this, std::placeholders::_1)
#define BIND_KEY_PRESSED_FUNCTION(x) std::bind(&Application::x, this, std::placeholders::_1)
#define BIND_KEY_RELEASED_FUNCTION(x) std::bind(&Application::x, this)
#define REGISTER_KEY_PRESSED_FUNCTION(key) m_keyPressed[#key[0]] = BIND_KEY_PRESSED_FUNCTION(_OnKeyPressed_##key)
#define REGISTER_KEY_RELEASED_FUNCTION(key) m_keyReleased[#key[0]] = BIND_KEY_RELEASED_FUNCTION(_OnKeyReleased_##key)

Application::Application()
    : m_running(true)
{
    m_window = std::unique_ptr<Window>(Window::Create());
    m_window->SetEventCallback(BIND_EVENT_CALLBACK(OnEvent));
    REGISTER_KEY_PRESSED_FUNCTION(a);
    REGISTER_KEY_PRESSED_FUNCTION(R);
    REGISTER_KEY_RELEASED_FUNCTION(q);
    REGISTER_KEY_RELEASED_FUNCTION(Q);
}

Application::~Application()
{
    CORE_TRACE("Application destructed.");
}

void Application::OnEvent(Event& e)
{
    CORE_TRACE("{0}", e);
    EventDispatcher ed(e);
    ed.Dispatch<WindowCloseEvent>(BIND_EVENT_CALLBACK(OnWindowClose));
//     ed.Dispatch<KeyPressedEvent>(BIND_EVENT_CALLBACK(OnKeyPressed));
    ed.Dispatch<KeyReleasedEvent>(BIND_EVENT_CALLBACK(OnKeyReleased));
}

bool Application::OnWindowClose(WindowCloseEvent& e)
{
    INFO("CLOSED");
    m_running = false;
    return true;
}

bool Application::OnKeyPressed(KeyPressedEvent& e)
{
    std::function<bool(int)> fn = m_keyPressed[e.GetKeyCode()];
    return fn == nullptr? false : fn(e.GetRepeatCount());
}

bool Application::OnKeyReleased(KeyReleasedEvent& e)
{
    std::function<bool()> fn = m_keyReleased[e.GetKeyCode()];
    return fn == nullptr? false : fn();
}

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
        m_window->OnUpdate();
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






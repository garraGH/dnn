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
Application::Application()
    : m_running(true)
{
    m_window = std::unique_ptr<Window>(Window::Create());
    m_window->SetEventCallback(BIND_EVENT_CALLBACK(OnEvent));
}

Application::~Application()
{

}

void Application::OnEvent(Event& e)
{
    CORE_TRACE("{0}", e);
    EventDispatcher ed(e);
    ed.Dispatch<WindowCloseEvent>(BIND_EVENT_CALLBACK(OnWindowClose));
    ed.Dispatch<KeyPressedEvent>(BIND_EVENT_CALLBACK(OnKeyPressed));
}

bool Application::OnWindowClose(WindowCloseEvent& e)
{
    INFO("CLOSED");
    m_running = false;
    return true;
}

bool Application::OnKeyPressed(KeyPressedEvent& e)
{
    if(e.GetKeyCode() == 'q')
    {
        INFO("QUIT");
        m_running = false;
    }
    return true;
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

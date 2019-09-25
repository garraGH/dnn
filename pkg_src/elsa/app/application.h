/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : /home/garra/study/dnn/pkg_src/elsa/app/application.h
* author      : Garra
* time        : 2019-09-24 10:43:40
* description : 
*
============================================*/


#pragma once
#include <memory>

#include "../event/event.h"
#include "../event/event_key.h"
#include "../event/event_mouse.h"
#include "../event/event_application.h"
#include "../window/window.h"

class Application
{
public:
    Application();
    virtual ~Application();

    virtual void Run();
protected:
    void OnEvent(Event& e);
    bool OnKeyPressed(KeyPressedEvent& e);
    bool OnWindowClose(WindowCloseEvent& e);

private:
    std::unique_ptr<Window> m_window;
    bool m_running;
};

Application* CreateApplication();

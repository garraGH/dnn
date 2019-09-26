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
#include <map>

#include "../event/event.h"
#include "../event/event_key.h"
#include "../event/event_mouse.h"
#include "../event/event_application.h"
#include "../window/window.h"
#include "../layer/layerstack.h"

#define ON_KEY_PRESSED(key) bool _OnKeyPressed_##key(int repeatCount)
#define ON_KEY_RELEASED(key) bool _OnKeyReleased_##key()

class Application
{
public:
    Application();
    virtual ~Application();

    virtual void Run();

    void PushLayer(Layer* layer);
    void PushOverlay(Layer* layer);

    inline Window& GetWindow() const { return *m_window; }
    inline static Application& Get() { return *s_instance; }

protected:
    void OnEvent(Event& e);
    bool OnKeyPressed(KeyPressedEvent& e);
    bool OnKeyReleased(KeyReleasedEvent& e);
    bool OnWindowClose(WindowCloseEvent& e);

private:
    ON_KEY_RELEASED(q);
    ON_KEY_RELEASED(Q);
    ON_KEY_PRESSED(R);
    ON_KEY_PRESSED(a);
    
    
    
private:
    std::unique_ptr<Window> m_window;
    bool m_running;
    std::map<int, std::function<bool(int)>> m_keyPressed;
    std::map<int, std::function<bool()>> m_keyReleased;
    LayerStack m_layerStack;

private:
    static Application* s_instance;
};

Application* CreateApplication();

/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : window_x11.h
* author      : Garra
* time        : 2019-09-25 17:51:43
* description : 
*
============================================*/


#pragma once
#include "window.h"
#include "array"
#include "../context/context.h"

class GLFWwindow;
class GLFWmonitor;
class X11Window : public Window
{                             
public:
    X11Window(const WindowsProps& props);
    virtual ~X11Window();

    void OnUpdate() override;
    void SwitchVSync() override;
    void SwitchFullscreen() override;
    bool IsVSync() const override { return m_data.bVSync; }
    bool IsFullscreen() const override { return m_data.bFullscreen; }
    void SetEventCallback(const EventCallback& eventCallback) override { m_data.eventCallback = eventCallback; }
    void UpdatePos() override;
    void UpdateSize() override;
    int* GetPos() override { return &m_data.pos[0]; }
    int* GetSize() override { return &m_data.size[0]; }
    void* GetNativeWindow() const override { return m_window; }

protected:
    void _Init(const WindowsProps& props);
    void _SaveProps(const WindowsProps& props);
    void _InitGLFW();
    void _InitGlad();
    void _InitGl3w();
    void _CreateContext();

    void _SetEventCallback();
    void _SetEventCallback_WindowResize();
    void _SetEventCallback_WindowRelocation();
    void _SetEventCallback_WindowClose();
    void _SetEventCallback_Key();
    void _SetEventCallback_Char();
    void _SetEventCallback_MouseButton();
    void _SetEventCallback_MouseScroll();
    void _SetEventCallback_MouseMove();

    void _Shutdown();

private:
    GraphicsContext* m_context = nullptr;
    GLFWwindow* m_window = nullptr;
    GLFWmonitor* m_moniter = nullptr;

//     std::array<int, 2> m_windowPos = { 0, 0 };
//     std::array<int, 2> m_windowSize = { 0, 0 };
//     std::array<int, 2> m_viewportSize = { 0, 0 };

    struct WindowData
    {
        std::array<int, 2> pos;
        std::array<int, 2> size;
        std::string title;
        bool bVSync = true;
        bool bFullscreen = false;
        EventCallback eventCallback;
    };
    WindowData m_data;
};

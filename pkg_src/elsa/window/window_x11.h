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
#include "glfw3.h"

class X11Window : public Window
{
public:
    X11Window(const WindowsProps& props);
    virtual ~X11Window();

    void OnUpdate() override;
    void SetVSync(bool enabled) override;
    void SetFullscreen(bool enabled) override;
    bool IsVSync() const override { return m_data.bVSync; }
    bool IsFullscreen() const override { return m_data.bFullscreen; }
    void SetEventCallback(const EventCallback& eventCallback) override { m_data.eventCallback = eventCallback; }
    inline unsigned int GetWidth() const override { return m_data.width; }
    inline unsigned int GetHeight() const override { return m_data.height; } 

protected:
    void _Init(const WindowsProps& props);
    void _SaveProps(const WindowsProps& props);
    void _InitGLFW();
    void _CreateWindow();
    void _SetEventCallback();
    void _Shutdown();

private:
    GLFWwindow* m_window;
    struct WindowData
    {
        std::string title;
        unsigned int width;
        unsigned int height;
        bool bVSync;
        bool bFullscreen;
        EventCallback eventCallback;
    };
    WindowData m_data;
};

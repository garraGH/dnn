/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : window.h
* author      : Garra
* time        : 2019-09-25 17:30:06
* description : 
*
============================================*/


#pragma once
#include "../event/event.h"

struct WindowsProps
{
    std::string title;
    int width;
    int height;
    WindowsProps(const std::string& t = "Elsa Engine", unsigned int w = 1280, unsigned int h = 720)
        : title(t)
        , width(w)
        , height(h)
    {

    }
};


class Window
{
public:
    using EventCallback = std::function<void(Event&)>;
    virtual ~Window() {}
    
    virtual void OnUpdate() = 0;
    virtual int* GetPos() = 0;
    virtual int* GetSize() = 0;
    virtual void UpdatePos() = 0;
    virtual void UpdateSize() = 0;
    
    virtual void SetEventCallback(const EventCallback& ecb) = 0;
    virtual void SwitchVSync() = 0;
    virtual void SwitchFullscreen() = 0;
    virtual bool IsVSync() const = 0;
    virtual bool IsFullscreen() const = 0;

    virtual void* GetNativeWindow() const = 0;

    static Window* Create(const WindowsProps& props = WindowsProps());
};

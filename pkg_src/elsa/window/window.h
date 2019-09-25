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
    unsigned int width;
    unsigned int height;
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
    virtual unsigned int GetWidth() const = 0;
    virtual unsigned int GetHeight() const = 0;
    
    virtual void SetEventCallback(const EventCallback& ecb) = 0;
    virtual void SetVSync(bool enabled) = 0;
    virtual void SetFullscreen(bool enabled) = 0;
    virtual bool IsVSync() const = 0;
    virtual bool IsFullscreen() const = 0;

    static Window* Create(const WindowsProps& props = WindowsProps());
};

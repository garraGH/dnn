/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : event_application.h
* author      : Garra
* time        : 2019-09-24 16:55:19
* description : 
*
============================================*/


#pragma once
#include "event.h"
#include <sstream>

class WindowResizeEvent : public Event
{
public:
    WindowResizeEvent(unsigned int width, unsigned int height) : m_width(width), m_height(height) {  }

    inline unsigned int GetWidth() const { return m_width; }
    inline unsigned int GetHeight() const { return m_height; }

    std::string ToString() const override
    {
        std::stringstream ss;
        ss << GetName() << ": " << m_width << ", " << m_height;
        return ss.str();
    }
    
    EVENT_CLASS_CATEGORY(EC_Application)
    EVENT_CLASS_TYPE(ET_WindowResize)

private:
    unsigned int m_width, m_height;
};

class WindowRelocationEvent : public Event
{
public:
    WindowRelocationEvent(int xpos, int ypos) : m_xpos(xpos), m_ypos(ypos) {  }

    inline int GetPosX() const { return m_xpos; }
    inline int GetPosY() const { return m_ypos; }

    std::string ToString() const override
    {
        std::stringstream ss;
        ss << GetName() << ": " << m_xpos << ", " << m_ypos;
        return ss.str();
    }
    
    EVENT_CLASS_CATEGORY(EC_Application)
    EVENT_CLASS_TYPE(ET_WindowRelocation)

private:
    int m_xpos, m_ypos;
};


class WindowMoveEvent : public Event
{
public:
    WindowMoveEvent(unsigned int posX, unsigned int posY) : m_posX(posX), m_posY(posY) {  }

    inline unsigned int GetPosX() const { return m_posX; }
    inline unsigned int GetPosY() const { return m_posY; }

    std::string ToString() const override
    {
        std::stringstream ss;
        ss << GetName() << ": " << m_posX << ", " << m_posY;
        return ss.str();
    }

    EVENT_CLASS_CATEGORY(EC_Application)
    EVENT_CLASS_TYPE(ET_WindowMove)

protected:
    unsigned int m_posX, m_posY;
};

class WindowFocusEvent : public Event
{
public:
    WindowFocusEvent() {  }

    EVENT_CLASS_CATEGORY(EC_Application)
    EVENT_CLASS_TYPE(ET_WindowFocus)
};

class WindowLostFocusEvent : public Event
{
public:
    WindowLostFocusEvent() {  }

    EVENT_CLASS_CATEGORY(EC_Application)
    EVENT_CLASS_TYPE(ET_WindowLostFocus)
};

class WindowCloseEvent : public Event
{
public:
    WindowCloseEvent() {  }

    EVENT_CLASS_CATEGORY(EC_Application)
    EVENT_CLASS_TYPE(ET_WindowClose)
};

class AppTickEvent : public Event
{
public:
    AppTickEvent() {  }

    EVENT_CLASS_CATEGORY(EC_Application)
    EVENT_CLASS_TYPE(ET_AppTick)
};

class AppUpdateEvent : public Event
{
public:
    AppUpdateEvent() {  }

    EVENT_CLASS_CATEGORY(EC_Application)
    EVENT_CLASS_TYPE(ET_AppUpdate)
};

class AppRenderEvent : public Event
{
public:
    AppRenderEvent() {  }

    EVENT_CLASS_CATEGORY(EC_Application)
    EVENT_CLASS_TYPE(ET_AppRender)
};


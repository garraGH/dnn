/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : event_mouse.h
* author      : Garra
* time        : 2019-09-24 16:40:18
* description : 
*
============================================*/


#pragma once
#include "event.h"
#include <sstream>

class MouseMovedEvent : public Event
{
public:
    MouseMovedEvent(float mouseX, float mouseY) : m_mouseX(mouseX), m_mouseY(mouseY) {  }

    inline float GetX() const { return m_mouseX; }
    inline float GetY() const { return m_mouseY; }

    std::string ToString() const override
    {
        std::stringstream ss;
        ss << GetName() << ": " << m_mouseX << ", " << m_mouseY;
        return ss.str();
    }

    EVENT_CLASS_CATEGORY(EC_Input|EC_Mouse)
    EVENT_CLASS_TYPE(ET_MouseMoved)
private:
    float m_mouseX, m_mouseY;
};

class MouseScrolledEvent : public Event
{
public:
    MouseScrolledEvent(float offsetX, float offsetY) : m_offsetX(offsetX), m_offsetY(offsetY) {  }
    
    inline float GetOffsetX() const { return m_offsetX; }
    inline float GetOffsetY() const { return m_offsetY; }

    std::string ToString() const override
    {
        std::stringstream ss;
        ss << GetName() << ": " << m_offsetX << ", " << m_offsetY;
        return ss.str();
    }

    EVENT_CLASS_CATEGORY(EC_Input|EC_Mouse)
    EVENT_CLASS_TYPE(ET_MouseScrolled)

private:
    float m_offsetX, m_offsetY;
};

class MosueButtonEvent : public Event
{
public:
    inline int GetMouseButton() const { return m_button; }

    EVENT_CLASS_CATEGORY(EC_Input|EC_Mouse|EC_MouseButton)
protected:
    MosueButtonEvent(int button) : m_button(button) {  }
    int m_button;
};


class MouseButtonPressedEvent : public MosueButtonEvent
{
public:
    MouseButtonPressedEvent(int button) : MosueButtonEvent(button) {  }
    std::string ToString() const override
    {
        std::stringstream ss;
        ss << GetName() << ": " << m_button;
        return ss.str();
    }

    EVENT_CLASS_TYPE(ET_MouseButtonPressed)
};

class MouseButtonReleasedEvent : public MosueButtonEvent
{
public:
    MouseButtonReleasedEvent(int button) : MosueButtonEvent(button) {  }
    std::string ToString() const override
    {
        std::stringstream ss;
        ss << GetName() << ": " << m_button;
        return ss.str();
    }

    EVENT_CLASS_TYPE(ET_MouseButtonReleased)
};

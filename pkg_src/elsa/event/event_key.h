/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : KeyEvent.h
* author      : Garra
* time        : 2019-09-24 16:21:58
* description : 
*
============================================*/


#pragma once
#include "event.h"
#include <sstream>

class KeyEvent : public Event
{
public:
    inline int GetKeyCode() const { return m_keyCode; }
    EVENT_CLASS_CATEGORY(EC_Keyboard|EC_Input);

protected:
    KeyEvent(int keyCode) : m_keyCode(keyCode) {  }
    int m_keyCode;
};

class KeyPressedEvent : public KeyEvent
{
public:
    KeyPressedEvent(int keyCode, int repeatCount) : KeyEvent(keyCode), m_repeatCount(repeatCount) {  }
    inline int GetRepeatCount() const { return m_repeatCount; }
    std::string ToString() const override
    {
        std::stringstream ss;
        ss  << GetName() << ": " << m_keyCode << " (" << m_repeatCount << " repeats)";
        return ss.str();
    }

    EVENT_CLASS_TYPE(ET_KeyPressed)

private:
    int m_repeatCount;
};

class KeyReleasedEvent : public KeyEvent
{
public:
    KeyReleasedEvent(int keyCode) : KeyEvent(keyCode) {  }
    std::string ToString() const override
    {
        std::stringstream ss;
        ss << GetName() << ": " << m_keyCode;
        return ss.str();
    }

    EVENT_CLASS_TYPE(ET_KeyReleased);
};

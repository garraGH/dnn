/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : event.h
* author      : Garra
* time        : 2019-09-24 15:00:56
* description : 
*
============================================*/


#pragma once
#include <string>
#include <ostream>
#include <functional>
#include "spdlog/fmt/ostr.h"

enum class EventType
{
    ET_UnKnown = 0, 
    ET_WindowClose, 
    ET_WindowResize, 
    ET_WindowFocus, 
    ET_WindowLostFocus, 
    ET_WindowMove, 
    ET_AppTick, 
    ET_AppUpdate, 
    ET_AppRender, 
    ET_KeyPressed, 
    ET_KeyReleased, 
    ET_MouseMoved, 
    ET_MouseScrolled,  
    ET_MouseButtonPressed, 
    ET_MouseButtonReleased, 
};


#define SETBIT(x) (1<<x)
enum EventCategory
{
    EC_UnKnown      = 0, 
    EC_Application  = SETBIT(0), 
    EC_Input        = SETBIT(1), 
    EC_Keyboard     = SETBIT(2), 
    EC_Mouse        = SETBIT(3), 
    EC_MouseButton  = SETBIT(4)
};

#define EVENT_CLASS_TYPE(type) \
    static EventType GetStaticType() { return EventType::type; } \
    virtual EventType GetType() const override { return GetStaticType(); } \
    virtual const char* GetName() const override { return #type; } 

#define EVENT_CLASS_CATEGORY(category) \
    virtual int GetCategoryFlags() const override { return category; }

class Event
{
    friend class EventDispatcher;
public:
    virtual EventType GetType() const = 0;
    virtual const char* GetName() const = 0;
    virtual int GetCategoryFlags() const = 0;
    virtual std::string ToString() const { return GetName(); }
    inline bool IsCategory(EventCategory c) { return GetCategoryFlags()&c; }
    inline bool IsHandled() const { return m_bHandled; }
    friend std::ostream& operator <<(std::ostream& out, const Event& e) { return out << e.ToString(); }
protected:
    bool m_bHandled = false;
};

class EventDispatcher
{
    template<typename T>
    using EventFn = std::function<bool(T&)>;
public:
    EventDispatcher(Event& e) : m_event(e) {}

    template<typename T>
    bool Dispatch(EventFn<T> func)
    {
        if(m_event.GetType() != T::GetStaticType())
        {
            return false;
        }
        m_event.m_bHandled = func(*(T*)&m_event);
        return true;
    }

private:
    Event& m_event;
};

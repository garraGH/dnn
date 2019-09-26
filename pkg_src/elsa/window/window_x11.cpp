/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : window_x11.cpp
* author      : Garra
* time        : 2019-09-25 18:06:25
* description : 
*
============================================*/

#include "core.h"
#include "glad.h"
#include "glfw3.h"
#include "window_x11.h"
#include "logger.h"
#include "../event/event_key.h"
#include "../event/event_mouse.h"
#include "../event/event_application.h"

static bool s_GLFWInitialized = false;
Window* Window::Create(const WindowsProps& props)
{
    return new X11Window(props);
}

static void GLFWErrorCallback(int error, const char* description)
{
    CORE_ERROR("GLFW Error ({0}): {1}", error, description);
}

X11Window::X11Window(const WindowsProps& props)
{
    _Init(props);
}

X11Window::~X11Window()
{
    _Shutdown();
}

void X11Window::_Init(const WindowsProps& props)
{
    _SaveProps(props);
    _InitGLFW();
    _CreateWindow();
    _InitGlad();
    _SetEventCallback();
    SetVSync(true);
}

void X11Window::_SaveProps(const WindowsProps& props)
{
    m_data.title = props.title;
    m_data.width = props.width;
    m_data.height = props.height;
    CORE_INFO("Creating X11Window {} ({}, {})", m_data.title, m_data.width, m_data.height);
}

void X11Window::_InitGLFW()
{
    if(!s_GLFWInitialized)
    {
        s_GLFWInitialized = true;
        int success = glfwInit();
        CORE_ASSERT(success, "Could not initialize GLFW!");
        glfwSetErrorCallback(GLFWErrorCallback);
    }
}

void X11Window::_InitGlad()
{
    int success = gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
    CORE_ASSERT(success, "Failed to initialized glad!");
}

void X11Window::_CreateWindow()
{
    m_window = glfwCreateWindow(m_data.width, m_data.height, m_data.title.c_str(), nullptr, nullptr);
    glfwMakeContextCurrent(m_window);
    glfwSetWindowUserPointer(m_window, &m_data);
}

void X11Window::_SetEventCallback()
{
    _SetEventCallback_WindowResize();
    _SetEventCallback_WindowClose();
    _SetEventCallback_Key();
    _SetEventCallback_MouseButton();
    _SetEventCallback_MouseMove();
    _SetEventCallback_MouseScroll();
}

void X11Window::_SetEventCallback_WindowResize()
{
    glfwSetWindowSizeCallback(m_window, [](GLFWwindow* window, int width, int height)
    {
       WindowData& data = *(WindowData*)glfwGetWindowUserPointer(window);
       WindowResizeEvent event(width, height);
       data.eventCallback(event);
       data.width = width;
       data.height = height;
    });
}

void X11Window::_SetEventCallback_WindowClose()
{
    glfwSetWindowCloseCallback(m_window, [](GLFWwindow* window)
    {
        WindowData& data = *(WindowData*)glfwGetWindowUserPointer(window);
        WindowCloseEvent event;
        data.eventCallback(event);
    });
}

void X11Window::_SetEventCallback_Key()
{
    glfwSetKeyCallback(m_window, [](GLFWwindow* window, int key, int scancode, int action, int mods)
    { 
        static int repeatCount = 0;
        WindowData& data = *(WindowData*)glfwGetWindowUserPointer(window);
        int keyCode = mods&GLFW_MOD_SHIFT? key : key+32;
        switch(action)
        {
            case GLFW_PRESS:
            {
                repeatCount = 0;
                KeyPressedEvent event(keyCode, repeatCount);
                data.eventCallback(event);
                break;
            }
            case GLFW_RELEASE:
            {
                KeyReleasedEvent event(keyCode);
                data.eventCallback(event);
                break;
            }
            case GLFW_REPEAT:
            {
                KeyPressedEvent event(keyCode, ++repeatCount);
                data.eventCallback(event);
                break;
            }
        }
    });
}

void X11Window::_SetEventCallback_MouseButton()
{
    glfwSetMouseButtonCallback(m_window, [](GLFWwindow* window, int button, int action, int mods)
    {
        WindowData& data = *(WindowData*)glfwGetWindowUserPointer(window);
        switch(action)
        {
            case GLFW_PRESS:
            {
                MouseButtonPressedEvent event(button);
                data.eventCallback(event);
                break;
            }
            case GLFW_RELEASE:
            {
                MouseButtonReleasedEvent event(button);
                data.eventCallback(event);
                break;
            }
        }
    });
}

void X11Window::_SetEventCallback_MouseScroll()
{
    glfwSetScrollCallback(m_window, [](GLFWwindow* window, double xOffset, double yOffset)
    { 
        WindowData& data = *(WindowData*)glfwGetWindowUserPointer(window);
        MouseScrolledEvent event(xOffset, yOffset);
        data.eventCallback(event);
    });
}

void X11Window::_SetEventCallback_MouseMove()
{
    glfwSetCursorPosCallback(m_window, [](GLFWwindow* window, double xPos, double yPos)
    { 
        WindowData& data = *(WindowData*)glfwGetWindowUserPointer(window);
        MouseMovedEvent event(xPos, yPos);
        data.eventCallback(event);
    });
}

void X11Window::_Shutdown()
{
    glfwDestroyWindow(m_window);
}


void X11Window::SetVSync(bool enabled)
{
    if(m_data.bVSync == enabled)
    {
        return ;
    }

    m_data.bVSync = enabled;
    glfwSwapInterval(enabled);
}

void X11Window::SetFullscreen(bool enabled)
{
    if(m_data.bFullscreen == enabled)
    {
        return ;
    }
    m_data.bFullscreen = enabled;
}


void X11Window::OnUpdate()
{
    glfwPollEvents()    ;
    glfwSwapBuffers(m_window);
}


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
#include "glad/gl.h"
// #include "GL/gl3w.h"
#include "GLFW/glfw3.h"
#include "window_x11.h"
#include "logger.h"
#include "../event/event_key.h"
#include "../event/event_mouse.h"
#include "../event/event_application.h"
#include "../context/context_opengl.h"

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
    _CreateContext();
//     _InitGl3w();
    _InitGlad();
    _SetEventCallback();
}

void X11Window::_SaveProps(const WindowsProps& props)
{
    m_data.title = props.title;
    m_data.size[0] = props.width;
    m_data.size[1] = props.height;
    CORE_INFO("Creating X11Window {} ({}, {})", m_data.title, m_data.size[0], m_data.size[1]);
}

void X11Window::_InitGLFW()
{
    if(!s_GLFWInitialized)
    {
        s_GLFWInitialized = true;
        int success = glfwInit();
        CORE_ASSERT(success, "Could not initialize GLFW!");
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
//         glfwWindowHint(GLFW_DECORATED, false);
        glfwSetErrorCallback(GLFWErrorCallback);
    }
}

void X11Window::_CreateContext()
{
    m_window = glfwCreateWindow(m_data.size[0], m_data.size[1], m_data.title.c_str(), nullptr, nullptr);
    m_moniter = glfwGetPrimaryMonitor();
    glfwGetWindowPos(m_window, &m_data.pos[0], &m_data.pos[1]);
    m_context = new OpenGLContext(m_window);
    m_context->Init();
    glfwSetWindowUserPointer(m_window, &m_data);
}

void X11Window::_InitGlad()
{
    int success = gladLoadGL();
    CORE_ASSERT(success, "Failed to initialized glad!");
}

void X11Window::_InitGl3w()
{
//     int success = gl3wInit();
//     CORE_ASSERT(success, "Failed to initialized gl3w!");
}

void X11Window::_SetEventCallback()
{
    _SetEventCallback_WindowResize();
    _SetEventCallback_WindowRelocation();
    _SetEventCallback_WindowClose();
    _SetEventCallback_Key();
    _SetEventCallback_Char();
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
       if(glfwGetPrimaryMonitor() == nullptr)
       {
           data.size[0] = width;
           data.size[1] = height;
       }
    });
}

void X11Window::_SetEventCallback_WindowRelocation()
{
    glfwSetWindowPosCallback(m_window, [](GLFWwindow* window, int xpos, int ypos)
    {
        WindowData& data = *(WindowData*)glfwGetWindowUserPointer(window);
        WindowRelocationEvent event(xpos, ypos);
        data.eventCallback(event);
        data.pos[0] = xpos;
        data.pos[1] = ypos;
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
        int keyCode = (0x41<=key&&key<=0x5A)&&(!(mods&GLFW_MOD_SHIFT))? key+32 : key;
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

void X11Window::_SetEventCallback_Char()
{
    glfwSetCharCallback(m_window, [](GLFWwindow* window, unsigned int keyCode)
    {
        WindowData& data = *(WindowData*)glfwGetWindowUserPointer(window);
        KeyTypedEvent event(keyCode);
        data.eventCallback(event);
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
                MouseButtonPressedEvent event(static_cast<MouseButtonCode>(button));
                data.eventCallback(event);
                break;
            }
            case GLFW_RELEASE:
            {
                MouseButtonReleasedEvent event(static_cast<MouseButtonCode>(button));
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
    glfwTerminate();
}


void X11Window::SwitchVSync()
{
    m_data.bVSync = !m_data.bVSync;
    glfwSwapInterval(m_data.bVSync);
}

void X11Window::SwitchFullscreen()
{
    m_data.bFullscreen = !m_data.bFullscreen;

    if(m_data.bFullscreen)
    {
        const GLFWvidmode* mode = glfwGetVideoMode(m_moniter);
        glfwSetWindowMonitor(m_window, m_moniter, 0, 0, mode->width, mode->height, 0);
    }
    else
    {
        glfwSetWindowMonitor(m_window, nullptr, m_data.pos[0], m_data.pos[1], m_data.size[0], m_data.size[1], 0);
    }
}

void X11Window::UpdatePos()
{
    glfwSetWindowPos(m_window, m_data.pos[0], m_data.pos[1]);
}

void X11Window::UpdateSize()
{
    glfwSetWindowSize(m_window, m_data.size[0], m_data.size[1]);
}

void X11Window::OnUpdate()
{
    glfwPollEvents();
    m_context->SwapBuffers();
}


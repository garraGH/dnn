/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : window_x11.cpp
* author      : Garra
* time        : 2019-09-25 18:06:25
* description : 
*
============================================*/


#include "window_x11.h"

static bool s_GLFWInitialized = false;
Window* Window::Create(const WindowsProps& props)
{
    return new X11Window(props);
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
    m_data.title = props.title;
    m_data.width = props.width;
    m_data.height = props.height;

    CORE_INFO("Creating X11Window {} ({}, {})", m_data.title, m_data.width, m_data.height);

    if(!s_GLFWInitialized)
    {
        s_GLFWInitialized = true;
        int success = glfwInit();
        CORE_ASSERT(success, "Could not initialize GLFW!");
    }

    m_window = glfwCreateWindow(m_data.width, m_data.height, m_data.title.c_str(), nullptr, nullptr);
    glfwMakeContextCurrent(m_window);
    glfwSetWindowUserPointer(m_window, &m_data);
    SetVSync(true);
}




void X11Window::_Shutdown()
{
    glfwDestroyWindow(m_window);
}




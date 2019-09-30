/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : context_opengl.cpp
* author      : Garra
* time        : 2019-09-30 16:51:10
* description : 
*
============================================*/


#include "context_opengl.h"
#include "glad/gl.h"
#include "GLFW/glfw3.h"
#include "../core.h"

OpenGLContext::OpenGLContext(GLFWwindow* window)
    : m_window(window)
{
    CORE_ASSERT(window, "Window handle is null!");
}


void OpenGLContext::Init()
{
    glfwMakeContextCurrent(m_window);
    int success = gladLoadGL(glfwGetProcAddress);
    CORE_ASSERT(success, "Failed to initialize GLAD!");

    CORE_INFO("OpenGL INFO:");
    CORE_INFO("     Vendor: {}", glGetString(GL_VENDOR));
    CORE_INFO("   Renderer: {}", glGetString(GL_RENDERER));
    CORE_INFO("    Version: {}", glGetString(GL_VERSION));
    CORE_INFO("       GLSL: {}", glGetString(GL_SHADING_LANGUAGE_VERSION));
}

void OpenGLContext::SwapBuffers()
{
    glfwSwapBuffers(m_window);
}

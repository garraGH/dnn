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
}

void OpenGLContext::SwapBuffers()
{
    glBegin(GL_TRIANGLES);
    glVertex2f(-0.5f, -0.5f);
    glVertex2f(+0.5f, -0.5f);
    glVertex2f(-0.0f, +0.5f);
    glEnd();
    glfwSwapBuffers(m_window);
}

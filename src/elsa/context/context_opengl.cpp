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
    int success = gladLoadGL();
    CORE_ASSERT(success, "Failed to initialize GLAD!");

    CORE_INFO("{}", "OpenGL INFO:");
    CORE_INFO("     Vendor: {}", glGetString(GL_VENDOR));
    CORE_INFO("   Renderer: {}", glGetString(GL_RENDERER));
    CORE_INFO("    Version: {}", glGetString(GL_VERSION));
    CORE_INFO("       GLSL: {}", glGetString(GL_SHADING_LANGUAGE_VERSION));

    int maxVertexAttribs = 0;
    int maxVertexUniformBlocks = 0;
    int maxVertexUniformComponents = 0;
    int maxFragmentUniformBlocks = 0;
    int maxFragmentUniformComponents = 0;
    glad_glGetIntegerv(GL_MAX_VERTEX_ATTRIBS, &maxVertexAttribs);
    glad_glGetIntegerv(GL_MAX_VERTEX_UNIFORM_BLOCKS, &maxVertexUniformBlocks);
    glad_glGetIntegerv(GL_MAX_VERTEX_UNIFORM_COMPONENTS, &maxVertexUniformComponents);
    glad_glGetIntegerv(GL_MAX_FRAGMENT_UNIFORM_BLOCKS, &maxFragmentUniformBlocks);
    glad_glGetIntegerv(GL_MAX_FRAGMENT_UNIFORM_COMPONENTS, &maxFragmentUniformComponents);

    CORE_INFO("            maxVertexAttribs: {}", maxVertexAttribs);
    CORE_INFO("      maxVertexUniformBlocks: {}", maxVertexUniformBlocks);
    CORE_INFO("  maxVertexUniformComponents: {}", maxVertexUniformComponents);
    CORE_INFO("    maxFragmentUniformBlocks: {}", maxFragmentUniformBlocks);
    CORE_INFO("maxFragmentUniformComponents: {}", maxFragmentUniformComponents);
    
}

void OpenGLContext::SwapBuffers()
{
    glfwSwapBuffers(m_window);
}

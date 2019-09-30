/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : context_opengl.h
* author      : Garra
* time        : 2019-09-30 16:51:10
* description : 
*
============================================*/


#pragma once

#include "context.h"

struct GLFWwindow;
class OpenGLContext : public GraphicsContext
{
public:
    OpenGLContext(GLFWwindow* window);

    virtual void Init() override;
    virtual void SwapBuffers() override;

private:
    GLFWwindow* m_window;
};

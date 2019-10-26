/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : src/elsa/renderer/api/api_opengl.cpp
* author      : Garra
* time        : 2019-10-03 17:45:14
* description : 
*
============================================*/


#include "api_opengl.h"
#include "glad/gl.h"

OpenGLAPI::OpenGLAPI()
{
    s_type = OpenGL;
}

void OpenGLAPI::SetViewport(const std::shared_ptr<Viewport>& viewport)
{
    auto [x, y, w, h] = viewport->GetRange();
    auto [r, g, b, a] = viewport->GetBackgroundColor();
    float depth = viewport->GetBackgroundDepth();
    glViewport(x, y, w, h);
    glScissor(x, y, w, h);
    glEnable(GL_SCISSOR_TEST);
    glClearDepth(depth);
    glClearColor(r, g, b, a);
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
    glDisable(GL_SCISSOR_TEST);
}

void OpenGLAPI::SetBackgroundColor(float r, float g, float b, float a)
{
    glClearColor(r, g, b, a);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void OpenGLAPI::DrawIndexed(const std::shared_ptr<BufferArray>& bufferArray)
{
    glDrawElements(GL_TRIANGLES, bufferArray->IndexCount(), bufferArray->IndexType(), nullptr);
}


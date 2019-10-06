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

void OpenGLAPI::SetBackgroundColor(float r, float g, float b, float a)
{
    glad_glClearColor(r, g, b, a);
    glad_glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void OpenGLAPI::DrawIndexed(const std::shared_ptr<BufferArray>& bufferArray)
{
    glad_glDrawElements(GL_TRIANGLES, bufferArray->IndexCount(), bufferArray->IndexType(), nullptr);
}



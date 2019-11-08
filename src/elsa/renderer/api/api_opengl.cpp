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

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glEnable(GL_DEPTH_WRITEMASK);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_DST_ALPHA);
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

void OpenGLAPI::SetPolygonMode(Renderer::PolygonMode mode)
{
    switch(mode)
    {
        case Renderer::PolygonMode::POINT: return glPolygonMode(GL_FRONT_AND_BACK, GL_POINT);
        case Renderer::PolygonMode::LINE:  return glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        case Renderer::PolygonMode::FILL:  return glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }
}

float OpenGLAPI::GetPixelDepth(int x, int y)
{
    float depth = 0;
    glad_glReadPixels(x, y, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &depth);
    return depth;
}

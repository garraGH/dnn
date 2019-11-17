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
    glEnable(GL_MULTISAMPLE);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_DST_ALPHA);
}

void OpenGLAPI::SetFrameBuffer(const std::shared_ptr<FrameBuffer>& frameBuffer)
{
    glBindFramebuffer(GL_FRAMEBUFFER, frameBuffer == nullptr? 0 : frameBuffer->ID());
}

void OpenGLAPI::BlitFrameBuffer(const std::shared_ptr<FrameBuffer>& from, const std::shared_ptr<FrameBuffer>& to)
{
    glBindFramebuffer(GL_READ_FRAMEBUFFER, from->ID());
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, to->ID());
    unsigned int w = from->GetWidth();
    unsigned int h = from->GetHeight();
    glBlitFramebuffer(0, 0, w, h, 0, 0, w, h, GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT|GL_STENCIL_BUFFER_BIT, GL_NEAREST);
}

void OpenGLAPI::SetBackground(const glm::vec4& color, float depth, float stencil)
{
    glClearColor(color.r, color.g, color.b, color.a);
    glClearDepth(depth);
    glClearStencil(stencil);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
}

void OpenGLAPI::DrawElements(const std::shared_ptr<BufferArray>& bufferArray, unsigned int nInstances)
{
    glDrawElementsInstanced(GL_TRIANGLES, bufferArray->IndexCount(), bufferArray->IndexType(), nullptr, nInstances);
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

float OpenGLAPI::GetPixelDepth(int x, int y, const std::shared_ptr<FrameBuffer>& frameBuffer)
{
    SetFrameBuffer(frameBuffer);

    float depth = 0;
    glReadPixels(x, y, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &depth);

    return depth;
}

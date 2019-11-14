/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : src/elsa/renderer/api/api_opengl.h
* author      : Garra
* time        : 2019-10-03 17:33:39
* description : 
*
============================================*/


#pragma once
#include "../renderer.h"

class OpenGLAPI : public Renderer::API
{
public:
    OpenGLAPI();
    
    virtual void SetViewport(const std::shared_ptr<Viewport>& viewport) override;
    virtual void SetFrameBuffer(const std::shared_ptr<FrameBuffer>& frameBuffer) override;
    virtual void SetBackgroundColor(float r, float g, float b, float a) override;
    virtual void DrawElements(const std::shared_ptr<BufferArray>& bufferArray, unsigned int nInstances) override;
    virtual void SetPolygonMode(Renderer::PolygonMode mode) override;
    virtual float GetPixelDepth(int x, int y, const std::shared_ptr<FrameBuffer>& frameBuffer) override;
};


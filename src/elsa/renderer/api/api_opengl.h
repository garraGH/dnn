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
    
    virtual void ResizeWindow(unsigned int width, unsigned int height) override;
    virtual void SetBackgroundColor(float r, float g, float b, float a) override;
    virtual void DrawIndexed(const std::shared_ptr<BufferArray>& bufferArray) override;
};


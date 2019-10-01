/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : buffer.cpp
* author      : Garra
* time        : 2019-10-01 22:13:28
* description : 
*
============================================*/


#include "buffer.h"
#include "../renderer.h"
#include "buffer_opengl.h"
#include "../../core.h"

Buffer* Buffer::CreateVertex(unsigned int size, float* data)
{
    switch(Renderer::GetAPI())
    {
        case RendererAPI::OpenGL:
            return new OpenGLVertexBuffer(size, data);
        default:
            CORE_ASSERT(false, "API is currently not supported!");
            return nullptr;
    }
}

Buffer* Buffer::CreateIndex(unsigned int size, unsigned int* data)
{
    switch(Renderer::GetAPI())
    {
        case RendererAPI::OpenGL:
            return new OpenGLIndexBuffer(size, data);
        default:
            CORE_ASSERT(false, "API is currently not supported!");
            return nullptr;
    }
}

unsigned int Buffer::GetCount() const 
{
    return m_count;
}

/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : buffer_opengl.h
* author      : Garra
* time        : 2019-10-01 22:13:39
* description : 
*
============================================*/


#pragma once
#include "buffer.h"

class OpenGLBuffer : public Buffer
{
public:
    OpenGLBuffer();
    virtual ~OpenGLBuffer();
};

class OpenGLVertexBuffer : public OpenGLBuffer
{
public:
    OpenGLVertexBuffer(unsigned int size, float* data);

    void Bind(unsigned int slot=0) const override;
    void Unbind() const override;
};


class OpenGLIndexBuffer : public OpenGLBuffer
{
public:
    OpenGLIndexBuffer(unsigned int size, unsigned int* data);
    void Bind(unsigned int slot=0) const override;
    void Unbind() const override;
};

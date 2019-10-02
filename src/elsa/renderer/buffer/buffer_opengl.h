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
#include "glad/gl.h"

class OpenGLBuffer : public Buffer
{
public:
    OpenGLBuffer(unsigned int size);
    virtual ~OpenGLBuffer();

protected:
    GLenum _TypeFrom(Element::DataType dataType) const;
};

class OpenGLVertexBuffer : public OpenGLBuffer
{
public:
    OpenGLVertexBuffer(unsigned int size, float* data);

    void Bind(unsigned int slot=0) const override;
    void Unbind() const override;

    virtual void ApplyLayout() const override;
};


class OpenGLIndexBuffer : public OpenGLBuffer
{
public:
    OpenGLIndexBuffer(unsigned int size, void* data);
    void Bind(unsigned int slot=0) const override;
    void Unbind() const override;
    GLenum GetIndexType();
};

class OpenGLBufferArray : public BufferArray
{
public:
    OpenGLBufferArray();
    virtual ~OpenGLBufferArray();
    
    virtual void Bind(unsigned int slot=0) const override;
    virtual void Unbind() const override;

    virtual void Add(std::shared_ptr<Buffer> buffer) override;
};

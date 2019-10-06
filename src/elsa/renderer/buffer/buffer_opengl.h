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
#include "../shader/shader.h"

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

    virtual void ApplyLayout(const std::shared_ptr<Shader>& shader) const override;
};


class OpenGLIndexBuffer : public OpenGLBuffer
{
public:
    OpenGLIndexBuffer(unsigned int size, void* data);
    void Bind(unsigned int slot=0) const override;
    void Unbind() const override;
    GLenum GetType();
};

class OpenGLBufferArray : public BufferArray
{
public:
    OpenGLBufferArray();
    virtual ~OpenGLBufferArray();

    virtual void Bind(const std::shared_ptr<Shader>& shader) override;
    
    virtual void Bind(unsigned int slot=0) const override;
    virtual void Unbind() const override;

    virtual void AddVertexBuffer(const std::shared_ptr<Buffer>& buffer) override;
    virtual void SetIndexBuffer(const std::shared_ptr<Buffer>& buffer) override;

    virtual unsigned int IndexCount() const override;
    virtual unsigned int IndexType() const override;

};

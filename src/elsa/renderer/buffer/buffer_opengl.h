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
    OpenGLBuffer(unsigned int size, const void* data);
    virtual ~OpenGLBuffer();

protected:
    GLenum _TypeFrom(Element::DataType dataType) const;
};

class OpenGLVertexBuffer : public OpenGLBuffer
{
public:
    OpenGLVertexBuffer(unsigned int size, const void* data);

    virtual void Bind(unsigned int slot=0) override;
    virtual void Unbind() const override;

    virtual void Bind(const std::shared_ptr<Shader>& shader) override;
};


class OpenGLIndexBuffer : public OpenGLBuffer
{
public:
    OpenGLIndexBuffer(unsigned int size, const void* data);
    virtual void Bind(unsigned int slot=0) override;
    virtual void Unbind() const override;
    GLenum GetType();
};

class OpenGLRenderBuffer : public RenderBuffer
{
public:
    OpenGLRenderBuffer(unsigned int width, unsigned int height, unsigned int samples=4, const std::string& name="unnamed");
    ~OpenGLRenderBuffer();

    virtual void Bind(unsigned int slot=0) override;
    virtual void Unbind() const override;

protected:
    virtual void _Reset() override;

private:
    void _Create();
    void _Delete();
};

class OpenGLFrameBuffer : public FrameBuffer
{
public:
    OpenGLFrameBuffer(unsigned int width, unsigned int height, unsigned int samples=4, const std::string& name="unnamed");
    ~OpenGLFrameBuffer();

    virtual void Bind(unsigned int slot=0) override;
    virtual void Unbind() const override;

protected:
    virtual void _Reset() override;

};

class OpenGLBufferArray : public BufferArray
{
public:
    OpenGLBufferArray();
    virtual ~OpenGLBufferArray();

    virtual void Bind(const std::shared_ptr<Shader>& shader) override;
    
    virtual void Bind(unsigned int slot=0) override;
    virtual void Unbind() const override;

    virtual void AddVertexBuffer(const std::shared_ptr<Buffer>& buffer) override;
    virtual void SetIndexBuffer(const std::shared_ptr<Buffer>& buffer) override;

    virtual unsigned int IndexCount() const override;
    virtual unsigned int IndexType() const override;

};

class OpenGLUniformBuffer : public UniformBuffer
{
public:
    OpenGLUniformBuffer(const std::string& name);
    ~OpenGLUniformBuffer();

    virtual void Upload(const std::string& name, const void* data) override;

    virtual void Bind(unsigned int slot=0) override;
    virtual void Unbind() const override;
     
protected:
    virtual void _Allocate() const override;
};

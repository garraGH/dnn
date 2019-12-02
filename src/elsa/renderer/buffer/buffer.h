/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : buffer.h
* author      : Garra
* time        : 2019-10-01 22:13:27
* description : 
*
============================================*/


#pragma once
#include <vector>
#include <string>
#include "../rendererobject.h"
#include "../../core.h"
#include "../shader/shader.h"
#include "../texture/texture.h"

class Buffer : public RenderObject, public std::enable_shared_from_this<Buffer>
{
public:
    class Element
    {
    public:
        enum class DataType { UnKnown = 0, Float, Float2, Float3, Float4, Mat3, Mat4, Int, Int2, Int3, Int4, UChar, UShort, UInt, Bool };

    public:
        Element(DataType type, const std::string& name="", bool normalized=false, unsigned int divisor=0);
        DataType Type() const { return m_type; }
        const std::string& Name() const { return m_name; } 
        bool Normalized() const { return m_normalized; }
        size_t Offset(unsigned int nthLocation=0) const;
        unsigned int Divisor() const { return m_divisor; }
        unsigned int Size() const ;
        unsigned int Components() const;
        unsigned int NumOfLocations() const;

    private:
        DataType m_type = DataType::UnKnown;
        std::string m_name;
        bool m_normalized = false;
        size_t m_offset = 0;
        unsigned int m_divisor = 0;
        friend class Buffer;
    };

    class Layout
    {
    public:
        Layout() {}
        Layout(const std::initializer_list<Element>& elements);
        std::vector<Element>::const_iterator begin() const { return  m_elements.begin(); }
        std::vector<Element>::const_iterator end() const { return m_elements.end(); }
        unsigned int Stride() const { return m_stride; }
        void Push(Element& element);
        bool Empty() const { return m_elements.empty(); }

    protected:
        void _calculateOffsetAndStride(Element& e);

    private:
        std::vector<Element> m_elements;
        unsigned int m_stride = 0;
    };
    
public:
    Buffer(unsigned int size, const void* data);
    virtual ~Buffer();

    virtual void Bind(unsigned int slot=0) override {}
    virtual void Unbind() const override {}

    unsigned int GetCount() const;
    std::shared_ptr<Buffer> SetLayout(const Layout& layout);
    virtual void Bind(const std::shared_ptr<Shader>& shader){}

    static std::shared_ptr<Buffer> CreateVertex(unsigned int size, const void* data);
    static std::shared_ptr<Buffer> CreateIndex(unsigned int size, const void* data);


protected:
    Layout m_layout;
    unsigned int m_size = 0;
    const void* m_data = nullptr;
};

class RenderBuffer : public RenderObject
{
public:
    RenderBuffer(unsigned int width, unsigned int height, unsigned int samples=4, const std::string& name="unnamed");
    void Reset(unsigned int width, unsigned int height, unsigned int samples=4);

    static std::shared_ptr<RenderBuffer> Create(unsigned int width, unsigned int height, unsigned int samples, const std::string& name);
protected: 
    virtual void _Reset() = 0;

protected:
    unsigned int m_width = 1920;
    unsigned int m_height = 1080;
    unsigned int m_samples = 4;
};

class FrameBuffer : public RenderObject
{
public:
    FrameBuffer(unsigned int width, unsigned int height, unsigned int samples=4, Texture::Format format=Texture::Format::RGB8, const std::string& name="unnamed");
    void Reset(unsigned int width, unsigned int height, unsigned int samples=4, Texture::Format format=Texture::Format::RGB8);

    unsigned int GetWidth() const { return m_width; }
    unsigned int GetHeight() const { return m_height; }
    unsigned int GetSamples() const { return m_samples; }
    Texture::Format GetFormat() const { return m_format; }

    const std::shared_ptr<Texture>& GetColorBuffer() const { return m_colorBuffer; }
    const std::shared_ptr<RenderBuffer>& GetDepthStencilBuffer() const { return m_depthStencilBuffer; }
    static std::shared_ptr<FrameBuffer> Create(unsigned int width, unsigned int height, unsigned int samples=4, Texture::Format format=Texture::Format::RGB8, const std::string& name="unnamed");

protected:
    virtual void _Reset() = 0;

protected:
    std::shared_ptr<Texture> m_colorBuffer = nullptr;
    std::shared_ptr<RenderBuffer> m_depthStencilBuffer = nullptr;

    unsigned int m_width = 1920;
    unsigned int m_height = 1080;
    unsigned int m_samples = 4;
    Texture::Format m_format = Texture::Format::RGB8;
};

class BufferArray : public RenderObject
{
public:

    virtual void Bind(const std::shared_ptr<Shader>& shader) = 0;

    virtual void AddVertexBuffer(const std::shared_ptr<Buffer>& buffer) = 0;
    virtual void SetIndexBuffer(const std::shared_ptr<Buffer>& buffer) = 0;

    virtual unsigned int IndexCount() const = 0;
    virtual unsigned int IndexType() const = 0;

    static std::shared_ptr<BufferArray> Create();

protected:
    std::shared_ptr<Shader> m_shader = nullptr;
    std::shared_ptr<Buffer> m_indexBuffer = nullptr;
    std::vector<std::shared_ptr<Buffer>> m_vertexBuffers;
};

class UniformBuffer : public RenderObject, public std::enable_shared_from_this<UniformBuffer>
{
public:
    UniformBuffer(const std::string& name) : RenderObject(name) {}
    std::shared_ptr<UniformBuffer> SetSize(int size);
    virtual void Upload(const std::string& name, const void* data) = 0;

    void Push(const std::string& name, const glm::ivec2& layout);

    static std::shared_ptr<UniformBuffer> Create(const std::string& name="unnamed");

protected:
    virtual void _Allocate() const = 0;

protected:
    int m_size = 0;
    std::map<std::string, glm::ivec2> m_layouts;
};

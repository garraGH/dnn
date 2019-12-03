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
    enum class Format
    {
        // Color
        R8, R8I, R8UI, R8_SNORM,                                                                                                                                        //   8 bit 
        RG8, RG8I, RG8UI, RG8_SNORM, R16, R16I, R16UI, R16_SNORM, R16F,                                                                                                 //  16 bit
        RGB8, RGB8I, RGB8UI, RGB8_SNORM, SRGB8,                                                                                                                         //  24 bit
        RGBA8, RGBA8I, RGBA8UI, RGBA8_SNORM, RG16, RG16I, RG16UI, RG16_SNORM, RG16F, R32I, R32UI, R32F, R11F_G11F_B10F, RGB10_A2, RGB10_A2UI, RGB9_E5, SRGB8_ALPHA8,    //  32 bit
        RGB16, RGB16I, RGB16UI, RGB16F, RGB16_SNORM,                                                                                                                    //  64 bit
        RGBA16, RGBA16I, RGBA16UI, RGBA16_SNORM, RGBA16F, RG32I, RG32UI, RG32F,                                                                                         //  96 bit
        RGB32I, RGB32UI, RGB32F,                                                                                                                                        // 128 bit
        RGBA32I, RGBA32UI, RGBA32F, 

        // Depth
        DEPTH_COMPONENT16, 
        DEPTH_COMPONENT24, 
        DEPTH_COMPONENT32, 
        DEPTH_COMPONENT32F, 

        // Stencil
        STENCIL_INDEX1, 
        STENCIL_INDEX4, 
        STENCIL_INDEX8, 
        STENCIL_INDEX16, 

        // DepthStencil
        DEPTH24_STENCIL8, 
        DEPTH32F_STENCIL8, 
    };

public:
    RenderBuffer(unsigned int width, unsigned int height, unsigned int samples=1, Format format=Format::R32F, const std::string& name="unnamed");
    void Reset(Format format);
    void Reset(unsigned int samples);
    void Reset(unsigned int width, unsigned int height);
    void Reset(unsigned int width, unsigned int height, unsigned int samples);
    void Reset(unsigned int width, unsigned int height, unsigned int samples, Format format);

    unsigned int GetWidth() const { return m_width; }
    unsigned int GetHeight() const { return m_height; }
    unsigned int GetSamples() const { return m_samples; }
    Format GetFormat() const { return m_format; }

    static std::shared_ptr<RenderBuffer> Create(unsigned int width, unsigned int height, unsigned int samples=1, Format format=Format::R32F, const std::string& name="unnamed");
protected: 
    virtual void _Reset() = 0;

protected:
    unsigned int m_width = 1920;
    unsigned int m_height = 1080;
    unsigned int m_samples = 1;
    Format m_format = Format::R32F;
};

class FrameBuffer : public RenderObject
{
public:
    FrameBuffer(unsigned int width, unsigned int height, unsigned int samples=1, const std::string& name="unnamed");
    void Reset(unsigned int width, unsigned int height, unsigned int samples=1);

    unsigned int GetWidth() const { return m_width; }
    unsigned int GetHeight() const { return m_height; }
    unsigned int GetSamples() const { return m_samples; }

    void AddColorBuffer(const std::string& name, Texture::Format format);
    void AddRenderBuffer(const std::string& name, RenderBuffer::Format format);

    const std::shared_ptr<Texture>& GetColorBuffer(const std::string& name) { return m_colorBuffers[name]; }
    const std::shared_ptr<RenderBuffer>& GetRenderBuffer(const std::string& name) { return m_renderBuffers[name]; }

    static std::shared_ptr<FrameBuffer> Create(unsigned int width, unsigned int height, unsigned int samples=1, const std::string& name="unnamed");

protected:
    virtual void _Reset() = 0;
    virtual void _Attach(const std::shared_ptr<Texture>& colorBuffer) = 0;
    virtual void _Attach(const std::shared_ptr<RenderBuffer>& renderBuffer) = 0;

protected:
    std::map< std::string, std::shared_ptr<Texture> > m_colorBuffers;
    std::map< std::string, std::shared_ptr<RenderBuffer> > m_renderBuffers;

    unsigned int m_width = 1920;
    unsigned int m_height = 1080;
    unsigned int m_samples = 1;
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

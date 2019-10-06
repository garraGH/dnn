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

class Buffer : public RenderObject
{
public:
    class Element
    {
    public:
        enum class DataType { UnKnown = 0, Float, Float2, Float3, Float4, Mat3, Mat4, Int, Int2, Int3, Int4, UChar, UShort, UInt, Bool };

    public:
        Element(DataType type, const std::string& name="", bool normalized=false);
        DataType Type() const { return m_type; }
        const std::string& Name() const { return m_name; } 
        bool Normalized() const { return m_normalized; }
        size_t Offset() const { return m_offset; }
        unsigned int Size() const ;
        unsigned int Components() const;

    private:
        DataType m_type = DataType::UnKnown;
        std::string m_name;
        bool m_normalized = false;
        size_t m_offset = 0;
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

    protected:
        void _calculateOffsetAndStride(Element& e);

    private:
        std::vector<Element> m_elements;
        unsigned int m_stride = 0;
    };
    
public:
    Buffer(unsigned int size, const void* data);
    virtual ~Buffer();

    virtual void Bind(unsigned int slot=0) const override {}
    virtual void Unbind() const override {}

    unsigned int GetCount() const;
    void SetLayout(const Layout& layout);
    virtual void Bind(const std::shared_ptr<Shader>& shader) const;

//     static Buffer* CreateVertex(unsigned int size, const void* data);
//     static Buffer* CreateIndex(unsigned int size, const void* data);

    static std::shared_ptr<Buffer> CreateVertex(unsigned int size, const void* data);
    static std::shared_ptr<Buffer> CreateIndex(unsigned int size, const void* data);


protected:
    Layout m_layout;
    unsigned int m_size = 0;
    const void* m_data = nullptr;
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


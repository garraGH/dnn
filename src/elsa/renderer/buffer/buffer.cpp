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

Buffer* Buffer::CreateIndex(unsigned int size, void* data)
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

Buffer::Buffer(unsigned int size)
    : m_size(size)
{

}

Buffer::~Buffer()
{

}

void Buffer::SetLayout(const Layout& layout)
{
    m_layout = layout;
    _ApplyLayout();
}

void Buffer::_ApplyLayout() const
{

}

unsigned int Buffer::GetCount() const 
{
    return m_size/m_layout.Stride();
}


Buffer::Element::Element(DataType type, bool normalized, const std::string& name)
    : m_type(type)
    , m_normalized(normalized)
    , m_name(name)
{
}

unsigned int Buffer::Element::Size() const 
{
    switch(m_type)
    {
        case DataType::Float:   return 1*4;
        case DataType::Float2:  return 2*4;
        case DataType::Float3:  return 3*4;
        case DataType::Float4:  return 4*4;
        case DataType::Int:     return 1*4;
        case DataType::Int2:    return 2*4;
        case DataType::Int3:    return 3*4;
        case DataType::Int4:    return 4*4;
        case DataType::UChar:   return 1*1;
        case DataType::UShort:  return 1*2;
        case DataType::UInt:    return 1*4;
        case DataType::Bool:    return 1*1;
        case DataType::Mat3:    return 3*3*4;
        case DataType::Mat4:    return 4*4*4;
        default: CORE_ASSERT(false, "UnKnown Buffer::DataType!"); return 0;
    }
}

unsigned int Buffer::Element::Components() const
{
    switch(m_type)
    {
        case DataType::Float:   return 1;
        case DataType::Float2:  return 2;
        case DataType::Float3:  return 3;
        case DataType::Float4:  return 4;
        case DataType::Int:     return 1;
        case DataType::Int2:    return 2;
        case DataType::Int3:    return 3;
        case DataType::Int4:    return 4;
        case DataType::UChar:   return 1;
        case DataType::UShort:  return 1;
        case DataType::UInt:    return 1;
        case DataType::Bool:    return 1;
        case DataType::Mat3:    return 3*3;
        case DataType::Mat4:    return 4*4;
        default: CORE_ASSERT(false, "UnKnown Buffer::DataType!"); return 0;
    }
}

Buffer::Layout::Layout(const std::initializer_list<Element>& elements)
{
    for(auto e : elements)
    {
        Push(e);
    }
}

void Buffer::Layout::Push(Element& element) 
{
    _calculateOffsetAndStride(element);
    m_elements.push_back(element);
}

void Buffer::Layout::_calculateOffsetAndStride(Element& e)
{
    e.m_offset = m_stride;
    m_stride += e.Size();
}


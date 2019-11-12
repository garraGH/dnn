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
#include "../../core.h"
#include "buffer_opengl.h"

std::shared_ptr<Buffer> Buffer::CreateVertex(unsigned int size, const void* data)
{
    switch(Renderer::GetAPIType())
    {
        case Renderer::API::OpenGL:
            return std::make_shared<OpenGLVertexBuffer>(size, data);
        default:
            CORE_ASSERT(false, "Buffer::CreateVertex: API is currently not supported!");
            return nullptr;
    }
}

std::shared_ptr<Buffer> Buffer::CreateIndex(unsigned int size, const void* data)
{
    switch(Renderer::GetAPIType())
    {
        case Renderer::API::OpenGL:
            return std::make_shared<OpenGLIndexBuffer>(size, data);
        default:
            CORE_ASSERT(false, "Buffer::CreateIndex: API is currently not supported!");
            return nullptr;
    }
}

std::shared_ptr<BufferArray> BufferArray::Create()
{
    switch(Renderer::GetAPIType())
    {
        case Renderer::API::OpenGL:
            return std::make_shared<OpenGLBufferArray>();
        default:
            CORE_ASSERT(false, "BufferArray::Create: API is currently not supported!");
            return nullptr;
    }
}


Buffer::Buffer(unsigned int size, const void* data)
    : m_size(size)
    , m_data(data)
{

}

Buffer::~Buffer()
{

}

std::shared_ptr<Buffer> Buffer::SetLayout(const Layout& layout)
{
    m_layout = layout;
    return shared_from_this();
}

unsigned int Buffer::GetCount() const 
{
    return m_size/m_layout.Stride();
}


Buffer::Element::Element(DataType type, const std::string& name, bool normalized, unsigned int divisor)
    : m_type(type)
    , m_name(name)
    , m_normalized(normalized)
    , m_divisor(divisor)
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
        case DataType::Mat3:    return 3;
        case DataType::Mat4:    return 4;
        default: CORE_ASSERT(false, "UnKnown Buffer::DataType!"); return 0;
    }
}

unsigned int Buffer::Element::NumOfLocations() const 
{
    switch(m_type)
    {
        case DataType::Mat3: return 3;
        case DataType::Mat4: return 4;
        default: return 1;
    }
}

size_t Buffer::Element::Offset(unsigned int nthLocation) const
{
    switch(m_type)
    {
        case DataType::Mat3: return m_offset+12*nthLocation;
        case DataType::Mat4: return m_offset+16*nthLocation;
        default: return m_offset;
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


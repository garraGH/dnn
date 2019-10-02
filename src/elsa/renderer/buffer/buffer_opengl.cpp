/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : buffer_opengl.cpp
* author      : Garra
* time        : 2019-10-01 22:13:42
* description : 
*
============================================*/


#include "buffer_opengl.h"
#include "../../core.h"

OpenGLBuffer::OpenGLBuffer(unsigned int size)
    : Buffer(size)
{
    glGenBuffers(1, &m_id);
}

OpenGLBuffer::~OpenGLBuffer()
{
    glDeleteBuffers(1, &m_id);
}

GLenum OpenGLBuffer::_TypeFrom(Element::DataType dataType) const
{
    switch(dataType)
    {
        case Element::DataType::Float:   return GL_FLOAT;
        case Element::DataType::Float2:  return GL_FLOAT;
        case Element::DataType::Float3:  return GL_FLOAT;
        case Element::DataType::Float4:  return GL_FLOAT;
        case Element::DataType::Int:     return GL_INT;
        case Element::DataType::Int2:    return GL_INT;
        case Element::DataType::Int3:    return GL_INT;
        case Element::DataType::Int4:    return GL_INT;
        case Element::DataType::UChar:   return GL_UNSIGNED_BYTE;
        case Element::DataType::UShort:  return GL_UNSIGNED_SHORT;
        case Element::DataType::UInt:    return GL_UNSIGNED_INT;
        case Element::DataType::Bool:    return GL_BOOL;
        case Element::DataType::Mat3:    return GL_FLOAT; 
        case Element::DataType::Mat4:    return GL_FLOAT; 
        default: CORE_ASSERT(false, "UnKnown Buffer::DataType!"); return -1;
    }
}

OpenGLVertexBuffer::OpenGLVertexBuffer(unsigned int size, float* data)
    : OpenGLBuffer(size)
{
    glBindBuffer(GL_ARRAY_BUFFER, m_id);
    glBufferData(GL_ARRAY_BUFFER, size, data, GL_STATIC_DRAW);
}

void OpenGLVertexBuffer::ApplyLayout() const
{
    unsigned int location = 0;
    for(const auto e : m_layout)
    {
        glEnableVertexAttribArray(location);
        glVertexAttribPointer(location, e.Components(), _TypeFrom(e.Type()), e.Normalized()? GL_TRUE:GL_FALSE, m_layout.Stride(), (const void*)e.Offset());
        location++;
    }
}

void OpenGLVertexBuffer::Bind(unsigned int slot) const
{
    glBindBuffer(GL_ARRAY_BUFFER, m_id);
}

void OpenGLVertexBuffer::Unbind() const
{
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

OpenGLIndexBuffer::OpenGLIndexBuffer(unsigned int size, void* data)
    : OpenGLBuffer(size)
{
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_id);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, size, data, GL_STATIC_DRAW);
}


void OpenGLIndexBuffer::Bind(unsigned int slot) const
{
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_id);
}

void OpenGLIndexBuffer::Unbind() const
{
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

GLenum OpenGLIndexBuffer::GetIndexType()
{
    const Element& e = *m_layout.begin();
    return _TypeFrom(e.Type());
}

OpenGLBufferArray::OpenGLBufferArray()
{
    glGenVertexArrays(1, &m_id);
}

OpenGLBufferArray::~OpenGLBufferArray()
{
    glDeleteVertexArrays(1, &m_id);
}

void OpenGLBufferArray::Bind(unsigned int slot) const
{
    glBindVertexArray(m_id);
}

void OpenGLBufferArray::Unbind() const
{
    glBindVertexArray(0);
}

void OpenGLBufferArray::Add(std::shared_ptr<Buffer> buffer)
{
    Bind();
    buffer->Bind();
    buffer->ApplyLayout();
}

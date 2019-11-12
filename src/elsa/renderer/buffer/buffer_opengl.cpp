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

OpenGLBuffer::OpenGLBuffer(unsigned int size, const void* data)
    : Buffer(size, data)
{
    glGenBuffers(1, &m_id);
    CORE_INFO("OpenGLBuffer size: {}, id: {}", size, m_id);
}

OpenGLBuffer::~OpenGLBuffer()
{
    glDeleteBuffers(1, &m_id);
    CORE_INFO("OpenGLBuffer delete: {}", m_id);
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

OpenGLVertexBuffer::OpenGLVertexBuffer(unsigned int size, const void* data)
    : OpenGLBuffer(size, data)
{
    glBindBuffer(GL_ARRAY_BUFFER, m_id);
    glBufferData(GL_ARRAY_BUFFER, size, data, GL_STATIC_DRAW);
}

void OpenGLVertexBuffer::Bind(const std::shared_ptr<Shader>& shader)
{
    Bind();
    int location = 0;
    for(const auto e : m_layout)
    {
        location = shader->GetLocation(e.Name());
        if(location == -1)
        {
            continue;
        }

        for(unsigned int i=0; i<e.NumOfLocations(); i++)
        {
//             INFO("{}, {}, {}, {}, {}", location, e.Name(), e.Components(), e.Divisor(), e.Offset(i));
            glEnableVertexAttribArray(location);
            glVertexAttribPointer(location, e.Components(), _TypeFrom(e.Type()), e.Normalized()? GL_TRUE:GL_FALSE, m_layout.Stride(), (const void*)e.Offset(i));
            glVertexAttribDivisor(location, e.Divisor());
            location++;
        }
    }
}

void OpenGLVertexBuffer::Bind(unsigned int slot)
{
    glBindBuffer(GL_ARRAY_BUFFER, m_id);
}

void OpenGLVertexBuffer::Unbind() const
{
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

OpenGLIndexBuffer::OpenGLIndexBuffer(unsigned int size, const void* data)
    : OpenGLBuffer(size, data)
{
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_id);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, size, data, GL_STATIC_DRAW);
}


void OpenGLIndexBuffer::Bind(unsigned int slot)
{
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_id);
}

void OpenGLIndexBuffer::Unbind() const
{
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

GLenum OpenGLIndexBuffer::GetType()
{
    CORE_ASSERT(!m_layout.Empty(), "OpenGLIndexBuffer::GetType: none layout!");
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

void OpenGLBufferArray::Bind(unsigned int slot)
{
    CORE_ASSERT(m_vertexBuffers.size()&&m_indexBuffer, "OpenGLBufferArray::Bind: Must have at least one VertexBuffer&&IndexBuffer to be Drawn!");
    glBindVertexArray(m_id);
    m_indexBuffer->Bind();
}

void OpenGLBufferArray::Unbind() const
{
    glBindVertexArray(0);
}

void OpenGLBufferArray::AddVertexBuffer(const std::shared_ptr<Buffer>& buffer)
{
    if(std::find(m_vertexBuffers.begin(), m_vertexBuffers.end(), buffer) != m_vertexBuffers.end())
    {
        return ;
    }
    m_vertexBuffers.push_back(buffer);
}

void OpenGLBufferArray::SetIndexBuffer(const std::shared_ptr<Buffer>& buffer)
{
    if(m_indexBuffer == buffer)
    {
        return ;
    }
    m_indexBuffer = buffer;
}

void OpenGLBufferArray::Bind(const std::shared_ptr<Shader>& shader)
{
    Bind();
    if(!shader || shader == m_shader)
    {
        return ;
    }
    m_shader = shader;

    for(const auto buffer : m_vertexBuffers)
    {
        buffer->Bind(shader);
    }
}

unsigned int OpenGLBufferArray::IndexCount() const
{
    CORE_ASSERT(m_indexBuffer, "OpenGLBufferArray::IndexCount: indexBuffer is nullptr!");
    return m_indexBuffer->GetCount();
}

unsigned int OpenGLBufferArray::IndexType() const
{
    return static_cast<OpenGLIndexBuffer*>(m_indexBuffer.get())->GetType();
}

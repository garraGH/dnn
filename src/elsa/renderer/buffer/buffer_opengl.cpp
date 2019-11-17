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
#include "../texture/texture2d.h"

OpenGLBuffer::OpenGLBuffer(unsigned int size, const void* data)
    : Buffer(size, data)
{
    glGenBuffers(1, &m_id);
    CORE_INFO("Create OpenGLBuffer id({}), size({})", m_id, m_size);
}

OpenGLBuffer::~OpenGLBuffer()
{
    glDeleteBuffers(1, &m_id);
    CORE_INFO("Delete OpenGLBuffer: id({}), size({})", m_id, m_size);
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

//////////////////////////////////////////////////////////////////////
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
//////////////////////////////////////////////////////////////////////
OpenGLRenderBuffer::OpenGLRenderBuffer(unsigned int width, unsigned int height, unsigned int samples, const std::string& name)
    : RenderBuffer(width, height, samples, name)
{
    _Create();
}

OpenGLRenderBuffer::~OpenGLRenderBuffer()
{
    _Delete();
}

void OpenGLRenderBuffer::_Create()
{
    glGenRenderbuffers(1, &m_id);
    glBindRenderbuffer(GL_RENDERBUFFER, m_id);
    m_samples == 1?
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, m_width, m_height) :
        glRenderbufferStorageMultisample(GL_RENDERBUFFER, m_samples, GL_DEPTH24_STENCIL8, m_width, m_height);

    CORE_INFO("Create OpenGLRenderBuffer id({}), size({}x{}), samples({})", m_id, m_width, m_height, m_samples);
}

void OpenGLRenderBuffer::_Delete()
{
    glDeleteRenderbuffers(1, &m_id);
    CORE_INFO("Delete OpenGLRenderBuffer id({}), size({}x{})", m_id, m_width, m_height);
}

void OpenGLRenderBuffer::_Reset()
{
    _Delete();
    _Create();
}

void OpenGLRenderBuffer::Bind(unsigned int slot)
{
    glBindRenderbuffer(GL_RENDERBUFFER, m_id);
}

void OpenGLRenderBuffer::Unbind() const
{
    glBindRenderbuffer(GL_RENDERBUFFER, 0);
}

//////////////////////////////////////////////////////////////////////
OpenGLFrameBuffer::OpenGLFrameBuffer(unsigned int width, unsigned int height, unsigned int samples, const std::string& name)
    : FrameBuffer(width, height, samples, name)
{
    glGenFramebuffers(1, &m_id);
    glBindFramebuffer(GL_FRAMEBUFFER, m_id);
    CORE_INFO("Create OpenGLFrameBuffer: id({}), size({}x{}), samples({})", m_id, m_width, m_height, m_samples);
    m_colorBuffer = Texture2D::Create(m_name+"_ColorBuffer")->Set(m_width, m_height, m_samples, Texture::Format::RGB8);
    m_depthStencilBuffer = RenderBuffer::Create(m_width, m_height, m_samples, m_name+"_DenpthStencilBuffer");
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, m_samples==1? GL_TEXTURE_2D:GL_TEXTURE_2D_MULTISAMPLE, m_colorBuffer->ID(), 0);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, m_depthStencilBuffer->ID());
    if(glCheckNamedFramebufferStatus(m_id, GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
    {
        CORE_ASSERT(false, "Create OpenGLFrameBuffer: FrameBuffer is not complete!");
    }
}

OpenGLFrameBuffer::~OpenGLFrameBuffer()
{
    glDeleteFramebuffers(1, &m_id);
    CORE_INFO("Delete OpenGLFrameBuffer id({}), size({}x{})", m_id, m_width, m_height);
}

void OpenGLFrameBuffer::Bind(unsigned int slot)
{
    glBindFramebuffer(GL_FRAMEBUFFER, m_id);
}

void OpenGLFrameBuffer::Unbind() const
{
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void OpenGLFrameBuffer::_Reset() 
{
    CORE_INFO("OpenGLFrameBuffer::_Reset: id({}), size({}x{}), samples({})", m_id, m_width, m_height, m_samples);
    m_colorBuffer->Reset(m_width, m_height, m_samples, Texture::Format::RGB8);
    m_depthStencilBuffer->Reset(m_width, m_height, m_samples);

    glNamedFramebufferTexture(m_id, GL_COLOR_ATTACHMENT0, m_colorBuffer->ID(), 0);
    glNamedFramebufferRenderbuffer(m_id, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, m_depthStencilBuffer->ID());
    if(glCheckNamedFramebufferStatus(m_id, GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
    {
        CORE_ASSERT(false, "OpengGLFrameBuffer::_Reset: FrameBuffer is not complete!");
    }
}



//////////////////////////////////////////////////////////////////////
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

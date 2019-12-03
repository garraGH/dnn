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
#include "glm/gtx/string_cast.hpp"

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
        location = shader->GetAttributeLocation(e.Name());
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
OpenGLRenderBuffer::OpenGLRenderBuffer(unsigned int width, unsigned int height, unsigned int samples, Format format, const std::string& name)
    : RenderBuffer(width, height, samples, format, name)
{
    _Create();
}

OpenGLRenderBuffer::~OpenGLRenderBuffer()
{
    _Destroy();
}

void OpenGLRenderBuffer::_Create()
{
    glGenRenderbuffers(1, &m_id);
    glBindRenderbuffer(GL_RENDERBUFFER, m_id);
    m_samples == 1?
        glRenderbufferStorage(GL_RENDERBUFFER, _Format(), m_width, m_height) :
        glRenderbufferStorageMultisample(GL_RENDERBUFFER, m_samples, _Format(), m_width, m_height);

    CORE_INFO("Create OpenGLRenderBuffer id({}), size({}x{}), samples({}), format({})", m_id, m_width, m_height, m_samples, _StringOfFormat());
}

void OpenGLRenderBuffer::_Destroy()
{
    glDeleteRenderbuffers(1, &m_id);
    CORE_INFO("Delete OpenGLRenderBuffer id({}), size({}x{})", m_id, m_width, m_height);
}

void OpenGLRenderBuffer::_Reset()
{
    _Destroy();
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

unsigned int OpenGLRenderBuffer::_Format() const
{
#define GLFormat(x) case Format::x: return GL_##x;
    switch(m_format)
    {
        GLFormat(R8);
        GLFormat(R8I);
        GLFormat(R8UI);
        GLFormat(R8_SNORM);

        GLFormat(RG8);
        GLFormat(RG8I);
        GLFormat(RG8UI);
        GLFormat(RG8_SNORM);

        GLFormat(R16);
        GLFormat(R16I);
        GLFormat(R16UI);
        GLFormat(R16_SNORM);
        GLFormat(R16F);

        GLFormat(RGB8);
        GLFormat(RGB8I);
        GLFormat(RGB8UI);
        GLFormat(RGB8_SNORM);
        GLFormat(SRGB8);

        GLFormat(RGBA8);
        GLFormat(RGBA8I);
        GLFormat(RGBA8UI);
        GLFormat(RGBA8_SNORM);

        GLFormat(RG16);
        GLFormat(RG16I);
        GLFormat(RG16UI);
        GLFormat(RG16_SNORM);
        GLFormat(RG16F);

        GLFormat(R32I);
        GLFormat(R32UI);
        GLFormat(R32F);

        GLFormat(R11F_G11F_B10F);
        GLFormat(RGB10_A2);
        GLFormat(RGB10_A2UI);
        GLFormat(RGB9_E5);
        GLFormat(SRGB8_ALPHA8);

        GLFormat(RGB16);
        GLFormat(RGB16I);
        GLFormat(RGB16UI);
        GLFormat(RGB16_SNORM);
        GLFormat(RGB16F);

        GLFormat(RG32I);
        GLFormat(RG32UI);
        GLFormat(RG32F);

        GLFormat(RGB32I);
        GLFormat(RGB32UI);
        GLFormat(RGB32F);

        GLFormat(RGBA32I);
        GLFormat(RGBA32UI);
        GLFormat(RGBA32F);

        GLFormat(DEPTH_COMPONENT16);
        GLFormat(DEPTH_COMPONENT24);
        GLFormat(DEPTH_COMPONENT32);
        GLFormat(DEPTH_COMPONENT32F);

        GLFormat(STENCIL_INDEX1);
        GLFormat(STENCIL_INDEX4);
        GLFormat(STENCIL_INDEX8);
        GLFormat(STENCIL_INDEX16);

        GLFormat(DEPTH24_STENCIL8);
        GLFormat(DEPTH32F_STENCIL8);

        default: CORE_ASSERT(false, "OpenGLRenderBuffer::_Format: UnKnown format!");
    }
#undef GLFormat
}


std::string OpenGLRenderBuffer::_StringOfFormat() const
{
#define StrFormat(x) case Format::x: return #x;
    switch(m_format)
    {
        StrFormat(R8);
        StrFormat(R8I);
        StrFormat(R8UI);
        StrFormat(R8_SNORM);

        StrFormat(RG8);
        StrFormat(RG8I);
        StrFormat(RG8UI);
        StrFormat(RG8_SNORM);

        StrFormat(R16);
        StrFormat(R16I);
        StrFormat(R16UI);
        StrFormat(R16_SNORM);
        StrFormat(R16F);

        StrFormat(RGB8);
        StrFormat(RGB8I);
        StrFormat(RGB8UI);
        StrFormat(RGB8_SNORM);
        StrFormat(SRGB8);

        StrFormat(RGBA8);
        StrFormat(RGBA8I);
        StrFormat(RGBA8UI);
        StrFormat(RGBA8_SNORM);

        StrFormat(RG16);
        StrFormat(RG16I);
        StrFormat(RG16UI);
        StrFormat(RG16_SNORM);
        StrFormat(RG16F);

        StrFormat(R32I);
        StrFormat(R32UI);
        StrFormat(R32F);

        StrFormat(R11F_G11F_B10F);
        StrFormat(RGB10_A2);
        StrFormat(RGB10_A2UI);
        StrFormat(RGB9_E5);
        StrFormat(SRGB8_ALPHA8);

        StrFormat(RGB16);
        StrFormat(RGB16I);
        StrFormat(RGB16UI);
        StrFormat(RGB16_SNORM);
        StrFormat(RGB16F);

        StrFormat(RG32I);
        StrFormat(RG32UI);
        StrFormat(RG32F);

        StrFormat(RGB32I);
        StrFormat(RGB32UI);
        StrFormat(RGB32F);

        StrFormat(RGBA32I);
        StrFormat(RGBA32UI);
        StrFormat(RGBA32F);

        StrFormat(DEPTH_COMPONENT16);
        StrFormat(DEPTH_COMPONENT24);
        StrFormat(DEPTH_COMPONENT32);
        StrFormat(DEPTH_COMPONENT32F);

        StrFormat(STENCIL_INDEX1);
        StrFormat(STENCIL_INDEX4);
        StrFormat(STENCIL_INDEX8);
        StrFormat(STENCIL_INDEX16);

        StrFormat(DEPTH24_STENCIL8);
        StrFormat(DEPTH32F_STENCIL8);
        
        default: return "UnkownFormat";
    }
#undef StrFormat
}

//////////////////////////////////////////////////////////////////////
OpenGLFrameBuffer::OpenGLFrameBuffer(unsigned int width, unsigned int height, unsigned int samples, const std::string& name)
    : FrameBuffer(width, height, samples, name)
{
    _Create();
}

void OpenGLFrameBuffer::_Create()
{
    CORE_INFO("Create OpenGLFrameBuffer: id({}), size({}x{}), samples({})", m_id, m_width, m_height, m_samples);
    glGenFramebuffers(1, &m_id);
}


void OpenGLFrameBuffer::_Check()
{
    if(glCheckNamedFramebufferStatus(m_id, GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
    {
        CORE_ASSERT(false, "Create OpenGLFrameBuffer: FrameBuffer is not complete!");
    }
}

OpenGLFrameBuffer::~OpenGLFrameBuffer()
{
    _Destroy();
}

void OpenGLFrameBuffer::_Destroy()
{
    CORE_INFO("Delete OpenGLFrameBuffer id({}), size({}x{})", m_id, m_width, m_height);
    glDeleteFramebuffers(1, &m_id);
}

void OpenGLFrameBuffer::Bind(unsigned int slot)
{
    glBindFramebuffer(GL_FRAMEBUFFER, m_id);
}

void OpenGLFrameBuffer::Unbind() const
{
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void OpenGLFrameBuffer::_ResetColorBuffers()
{
    for(auto [name, cb] : m_colorBuffers)
    {
        cb->Reset(m_width, m_height, m_samples);
        _Attach(cb);
    }
}

void OpenGLFrameBuffer::_ResetRenderBuffers()
{
    for(auto [name, rb] : m_renderBuffers)
    {
        rb->Reset(m_width, m_height, m_samples);
        _Attach(rb);
    }
}

void OpenGLFrameBuffer::_Reset() 
{
    CORE_INFO("OpenGLFrameBuffer::_Reset: id({}), size({}x{}), samples({})", m_id, m_width, m_height, m_samples);
    _ResetColorBuffers();
    _ResetRenderBuffers();
    _Check();
//     m_colorBuffer->Reset(m_width, m_height, m_samples, m_format);
//     m_depthStencilBuffer->Reset(m_width, m_height, m_samples);
// 
//     glNamedFramebufferTexture(m_id, GL_COLOR_ATTACHMENT0, m_colorBuffer->ID(), 0);
//     glNamedFramebufferRenderbuffer(m_id, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, m_depthStencilBuffer->ID());
//     if(glCheckNamedFramebufferStatus(m_id, GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
//     {
//         CORE_ASSERT(false, "OpengGLFrameBuffer::_Reset: FrameBuffer is not complete!");
//     }
}

void OpenGLFrameBuffer::_Attach(const std::shared_ptr<Texture>& colorBuffer)
{
    glBindFramebuffer(GL_FRAMEBUFFER, m_id);

    int n = m_colorBuffers.size();
    glNamedFramebufferTexture(m_id, GL_COLOR_ATTACHMENT0+n-1, colorBuffer->ID(), 0);
    _Check();

    unsigned int* attachments = new unsigned int[n];
    for(int i=0; i<n; i++)
    {
        attachments[i] = GL_COLOR_ATTACHMENT0+i;
    }
    glDrawBuffers(n, attachments);
    delete[] attachments;

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

unsigned int OpenGLFrameBuffer::_Attachment(RenderBuffer::Format format) const
{
    switch(format)
    {
        case RenderBuffer::Format::R8:
        case RenderBuffer::Format::R16:
        case RenderBuffer::Format::RGB8:
        case RenderBuffer::Format::RG8:
        case RenderBuffer::Format::RG16:
        case RenderBuffer::Format::RGB16:
        case RenderBuffer::Format::RGBA8:
        case RenderBuffer::Format::SRGB8:
        case RenderBuffer::Format::RGBA16:
        case RenderBuffer::Format::R8I:
        case RenderBuffer::Format::R16F:
        case RenderBuffer::Format::R16I:
        case RenderBuffer::Format::R32F:
        case RenderBuffer::Format::R32I:
        case RenderBuffer::Format::R8UI:
        case RenderBuffer::Format::RG8I:
        case RenderBuffer::Format::R16UI:
        case RenderBuffer::Format::R32UI:
        case RenderBuffer::Format::RG16F:
        case RenderBuffer::Format::RG16I:
        case RenderBuffer::Format::RG32F:
        case RenderBuffer::Format::RG32I:
        case RenderBuffer::Format::RG8UI:
        case RenderBuffer::Format::RGB8I:
        case RenderBuffer::Format::RG16UI:
        case RenderBuffer::Format::RG32UI:
        case RenderBuffer::Format::RGB16F:
        case RenderBuffer::Format::RGB16I:
        case RenderBuffer::Format::RGB32F:
        case RenderBuffer::Format::RGB32I:
        case RenderBuffer::Format::RGB8UI:
        case RenderBuffer::Format::RGBA8I:
        case RenderBuffer::Format::RGB16UI:
        case RenderBuffer::Format::RGB32UI:
        case RenderBuffer::Format::RGB9_E5:
        case RenderBuffer::Format::RGBA16F:
        case RenderBuffer::Format::RGBA16I:
        case RenderBuffer::Format::RGBA32F:
        case RenderBuffer::Format::RGBA32I:
        case RenderBuffer::Format::RGBA8UI:
        case RenderBuffer::Format::R8_SNORM:
        case RenderBuffer::Format::RGB10_A2:
        case RenderBuffer::Format::RGBA16UI:
        case RenderBuffer::Format::RGBA32UI:
        case RenderBuffer::Format::R16_SNORM:
        case RenderBuffer::Format::RG8_SNORM:
        case RenderBuffer::Format::RG16_SNORM:
        case RenderBuffer::Format::RGB8_SNORM:
        case RenderBuffer::Format::RGB16_SNORM:
        case RenderBuffer::Format::RGBA8_SNORM:
        case RenderBuffer::Format::RGBA16_SNORM:
        case RenderBuffer::Format::SRGB8_ALPHA8:
        case RenderBuffer::Format::R11F_G11F_B10F:
        case RenderBuffer::Format::RGB10_A2UI: 
            return GL_COLOR_ATTACHMENT0;
        case RenderBuffer::Format::STENCIL_INDEX1:
        case RenderBuffer::Format::STENCIL_INDEX4:
        case RenderBuffer::Format::STENCIL_INDEX8:
        case RenderBuffer::Format::STENCIL_INDEX16:
            return GL_STENCIL_ATTACHMENT;
        case RenderBuffer::Format::DEPTH_COMPONENT16:
        case RenderBuffer::Format::DEPTH_COMPONENT24:
        case RenderBuffer::Format::DEPTH_COMPONENT32:
        case RenderBuffer::Format::DEPTH_COMPONENT32F:
            return GL_DEPTH_ATTACHMENT;
        case RenderBuffer::Format::DEPTH24_STENCIL8:
        case RenderBuffer::Format::DEPTH32F_STENCIL8:
            return GL_DEPTH_STENCIL_ATTACHMENT;
        default:
            CORE_ASSERT(false, "OpenGLFrameBuffer::_Attachment: UnkownFormat.");
            return -1;
    }
}

void OpenGLFrameBuffer::_Attach(const std::shared_ptr<RenderBuffer>& renderBuffer)
{
    glBindFramebuffer(GL_FRAMEBUFFER, m_id);

    glNamedFramebufferRenderbuffer(m_id, _Attachment(renderBuffer->GetFormat()), GL_RENDERBUFFER, renderBuffer->ID());
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
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

/////////////////////////////////////////////////////////////////////////////////////
OpenGLUniformBuffer::OpenGLUniformBuffer(const std::string& name)
    : UniformBuffer(name)
{
    glGenBuffers(1, &m_id);
}

OpenGLUniformBuffer::~OpenGLUniformBuffer()
{
    glDeleteBuffers(1, &m_id);
}


void OpenGLUniformBuffer::Bind(unsigned int slot)
{
    glBindBuffer(GL_UNIFORM_BUFFER, m_id);
}

void OpenGLUniformBuffer::Unbind() const
{
    glBindBuffer(GL_UNIFORM_BUFFER, 0);
}

void OpenGLUniformBuffer::Upload(const std::string& name, const void* data)
{
//     INFO("UniformBuffer({})::UpLoad: {}", m_name, name);
//     for(int i=0; i<m_layouts[name].y/16; i++)
//     {
//         INFO("{}", glm::to_string(*((glm::vec4*)data+i)));
//     }
    glBindBuffer(GL_UNIFORM_BUFFER, m_id);
    glBufferSubData(GL_UNIFORM_BUFFER, m_layouts[name].x, m_layouts[name].y, data);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);
}

void OpenGLUniformBuffer::_Allocate() const
{
    glBindBuffer(GL_UNIFORM_BUFFER, m_id);
    glBufferData(GL_UNIFORM_BUFFER, m_size, nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);
}

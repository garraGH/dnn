/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : texture2d_opengl.cpp
* author      : Garra
* time        : 2019-10-14 20:08:04
* description : 
*
============================================*/
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "texture2d_opengl.h"
#include "glad/gl.h"
#include "../../core.h"

std::shared_ptr<Texture2D> OpenGLTexture2D::Create(const std::string& name)
{
    return std::make_shared<OpenGLTexture2D>(name);
}

OpenGLTexture2D::OpenGLTexture2D(const std::string& name)
    : Texture2D(name)
{
    _Create();
}

OpenGLTexture2D::~OpenGLTexture2D()
{
    _Destroy();
}

void OpenGLTexture2D::_Create()
{
    glGenTextures(1, &m_id);
    INFO("Create OpenGLTexture2D: {}", m_id);
}

void OpenGLTexture2D::_Destroy()
{
    INFO("Destroy OpenGLTexture2D: {}, {}x{}", m_id, m_width, m_height);
    glDeleteTextures(1, &m_id);
}

void OpenGLTexture2D::_Load()
{
    stbi_set_flip_vertically_on_load(true);
    stbi_uc* data = stbi_load(m_imagePath.c_str(), (int*)&m_width, (int*)&m_height, (int*)&m_channel, 0);
    CORE_ASSERT(data, "OpenGLTexture2D::Load: Failed to load image: "+m_imagePath);

    GLenum baseFormat = GL_RGB;
    GLenum sizedFormat = GL_RGB8;
    if(m_channel == 1)
    {
        baseFormat = GL_RED;
        sizedFormat = stbi_is_16_bit(m_imagePath.c_str())? GL_R16 : GL_R8;
    }
    else if(m_channel == 3)
    {
        baseFormat = GL_RGB;
        sizedFormat = GL_RGB8;
    }
    else if(m_channel == 4)
    {
        baseFormat = GL_RGBA;
        sizedFormat = GL_RGBA8;
    }
    else
    {
        CORE_ASSERT(false, "OpenGLTexture2D::_Load: Unsupported channel-"+std::to_string(m_channel));
    }
    glBindTexture(GL_TEXTURE_2D, m_id);
    glTextureParameteri(m_id, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTextureParameteri(m_id, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTextureStorage2D(m_id, 1, sizedFormat, m_width, m_height);
    glTextureSubImage2D(m_id, 0, 0, 0, m_width, m_height, baseFormat, GL_UNSIGNED_BYTE, data);

    stbi_image_free(data);

    INFO("OpenGLTexture2D::_Load: {}, {}x{}x{}", m_imagePath, m_width, m_height, m_channel);
}

void OpenGLTexture2D::_Allocate()
{
    INFO("OpenGLTexture2D::_Allocate: size({}x{}), samples({})", m_width, m_height, m_samples);
    m_samples == 1? 
        glTextureStorage2D(m_id, 1, _OpenGLFormat(), m_width, m_height) :
        glTextureStorage2DMultisample(m_id, m_samples, _OpenGLFormat(), m_width, m_height, GL_TRUE);
    if(m_samples == 1)
    {
        glBindTexture(GL_TEXTURE_2D, m_id);
        glTextureStorage2D(m_id, 1, _OpenGLFormat(), m_width, m_height);
        glBindTexture(GL_TEXTURE_2D, 0);
    }
    else
    {
        glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, m_id);
        glTextureStorage2DMultisample(m_id, m_samples, _OpenGLFormat(), m_width, m_height, GL_TRUE);
        glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, 0);
    }
}

void OpenGLTexture2D::Bind(unsigned int slot)
{
    m_slot = slot;

    glBindTextureUnit(slot, m_id);
}

void OpenGLTexture2D::Unbind() const
{
    glBindTextureUnit(m_slot, 0);
}

unsigned int OpenGLTexture2D::_OpenGLFormat()
{
    switch(m_format)
    {
        case Format::R8:        return GL_R8;
        case Format::R8SN:      return GL_R8_SNORM;
        case Format::R16:       return GL_R16;
        case Format::R16SN:     return GL_R16_SNORM;
        case Format::RG8:       return GL_RG8;
        case Format::RG8SN:     return GL_RG8_SNORM;
        case Format::RG16:      return GL_RG16;
        case Format::RG16SN:    return GL_RG16_SNORM;
        case Format::RG3B2:     return GL_R3_G3_B2;
        case Format::RGB4:      return GL_RGB4;
        case Format::RGB5:      return GL_RGB5;
        case Format::RGB8:      return GL_RGB8;
        case Format::RGB8SN:    return GL_RGB8_SNORM;
        case Format::RGB10:     return GL_RGB10;
        case Format::RGB12:     return GL_RGB12;
        case Format::RGB16SN:   return GL_RGB16_SNORM;
        case Format::RGBA2:     return GL_RGBA2;
        case Format::RGBA4:     return GL_RGBA4;
        case Format::RGB5A1:    return GL_RGB5_A1;
        case Format::RGBA8:     return GL_RGBA8;
        case Format::RGBA8SN:   return GL_RGBA8_SNORM;
        case Format::RGB10A2:   return GL_RGB10_A2;
        case Format::RGB10A2UI: return GL_RGB10_A2UI;
        case Format::RGBA12:    return GL_RGBA12;
        case Format::RGBA16:    return GL_RGBA16;
        case Format::SRGB8:     return GL_SRGB8;
        case Format::SRGBA8:    return GL_SRGB8_ALPHA8;
        case Format::R16F:      return GL_R16F;
        case Format::RG16F:     return GL_RG16F;
        case Format::RGB16F:    return GL_RGB16F;
        case Format::RGBA16F:   return GL_RGBA16F;
        case Format::R32F:      return GL_R32F;
        case Format::RG32F:     return GL_RG32F;
        case Format::RGB32F:    return GL_RGB32F;
        case Format::RGBA32F:   return GL_RGBA32F;
        case Format::RG11B10F:  return GL_R11F_G11F_B10F;
        case Format::RGB9E5:    return GL_RGB9_E5;
        case Format::R8I:       return GL_R8I;
        case Format::R8UI:      return GL_R8UI;
        case Format::R16I:      return GL_R16I;
        case Format::R16UI:     return GL_R16UI;
        case Format::R32I:      return GL_R32I;
        case Format::R32UI:     return GL_R32UI;
        case Format::RG8I:      return GL_RG8I;
        case Format::RG8UI:     return GL_RG8UI;
        case Format::RG16I:     return GL_RG16I;
        case Format::RG16UI:    return GL_RG16UI;
        case Format::RG32I:     return GL_RG32I;
        case Format::RG32UI:    return GL_RG32UI;
        case Format::RGB8I:     return GL_RGB8I;
        case Format::RGB8UI:    return GL_RGB8UI;
        case Format::RGB16I:    return GL_RGB16I;
        case Format::RGB16UI:   return GL_RGB16UI;
        case Format::RGB32I:    return GL_RGB32I;
        case Format::RGB32UI:   return GL_RGB32UI;
        case Format::RGBA8I:    return GL_RGBA8I;
        case Format::RGBA8UI:   return GL_RGBA8UI;
        case Format::RGBA16I:   return GL_RGBA16I;
        case Format::RGBA16UI:  return GL_RGBA16UI;
        case Format::RGBA32I:   return GL_RGBA32I;
        case Format::RGBA32UI:  return GL_RGBA32UI;

        default: CORE_ASSERT(false, "OpenGLTexture2D::_OpenGLFormat: Unkown format!"); return 0;
    }
}

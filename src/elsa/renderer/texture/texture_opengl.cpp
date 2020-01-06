/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : src/elsa/renderer/texture/texture_opengl.cpp
* author      : Garra
* time        : 2019-12-14 13:49:25
* description : 
*
============================================*/


#include "texture_opengl.h"
#include "glad/gl.h"
#include "core.h"


void OpenGLTexture::GenerateMipmap()
{
    GLCheck(glGenerateTextureMipmap(m_id));
    GLCheck(glTextureParameteri(m_id, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR));
}

void OpenGLTexture::_Create()
{
    glGenTextures(1, &m_id);
}

void OpenGLTexture::_Destroy()
{
    glDeleteTextures(1, &m_id);
}

unsigned int OpenGLTexture::_FormatOfGraphicsAPI()
{
    switch(m_format)
    {
        case Texture::Format::R8:        return GL_R8;
        case Texture::Format::R8SN:      return GL_R8_SNORM;
        case Texture::Format::R16:       return GL_R16;
        case Texture::Format::R16SN:     return GL_R16_SNORM;
        case Texture::Format::RG8:       return GL_RG8;
        case Texture::Format::RG8SN:     return GL_RG8_SNORM;
        case Texture::Format::RG16:      return GL_RG16;
        case Texture::Format::RG16SN:    return GL_RG16_SNORM;
        case Texture::Format::RG3B2:     return GL_R3_G3_B2;
        case Texture::Format::RGB4:      return GL_RGB4;
        case Texture::Format::RGB5:      return GL_RGB5;
        case Texture::Format::RGB8:      return GL_RGB8;
        case Texture::Format::RGB8SN:    return GL_RGB8_SNORM;
        case Texture::Format::RGB10:     return GL_RGB10;
        case Texture::Format::RGB12:     return GL_RGB12;
        case Texture::Format::RGB16SN:   return GL_RGB16_SNORM;
        case Texture::Format::RGBA2:     return GL_RGBA2;
        case Texture::Format::RGBA4:     return GL_RGBA4;
        case Texture::Format::RGB5A1:    return GL_RGB5_A1;
        case Texture::Format::RGBA8:     return GL_RGBA8;
        case Texture::Format::RGBA8SN:   return GL_RGBA8_SNORM;
        case Texture::Format::RGB10A2:   return GL_RGB10_A2;
        case Texture::Format::RGB10A2UI: return GL_RGB10_A2UI;
        case Texture::Format::RGBA12:    return GL_RGBA12;
        case Texture::Format::RGBA16:    return GL_RGBA16;
        case Texture::Format::SRGB8:     return GL_SRGB8;
        case Texture::Format::SRGBA8:    return GL_SRGB8_ALPHA8;
        case Texture::Format::R16F:      return GL_R16F;
        case Texture::Format::RG16F:     return GL_RG16F;
        case Texture::Format::RGB16F:    return GL_RGB16F;
        case Texture::Format::RGBA16F:   return GL_RGBA16F;
        case Texture::Format::R32F:      return GL_R32F;
        case Texture::Format::RG32F:     return GL_RG32F;
        case Texture::Format::RGB32F:    return GL_RGB32F;
        case Texture::Format::RGBA32F:   return GL_RGBA32F;
        case Texture::Format::RG11B10F:  return GL_R11F_G11F_B10F;
        case Texture::Format::RGB9E5:    return GL_RGB9_E5;
        case Texture::Format::R8I:       return GL_R8I;
        case Texture::Format::R8UI:      return GL_R8UI;
        case Texture::Format::R16I:      return GL_R16I;
        case Texture::Format::R16UI:     return GL_R16UI;
        case Texture::Format::R32I:      return GL_R32I;
        case Texture::Format::R32UI:     return GL_R32UI;
        case Texture::Format::RG8I:      return GL_RG8I;
        case Texture::Format::RG8UI:     return GL_RG8UI;
        case Texture::Format::RG16I:     return GL_RG16I;
        case Texture::Format::RG16UI:    return GL_RG16UI;
        case Texture::Format::RG32I:     return GL_RG32I;
        case Texture::Format::RG32UI:    return GL_RG32UI;
        case Texture::Format::RGB8I:     return GL_RGB8I;
        case Texture::Format::RGB8UI:    return GL_RGB8UI;
        case Texture::Format::RGB16I:    return GL_RGB16I;
        case Texture::Format::RGB16UI:   return GL_RGB16UI;
        case Texture::Format::RGB32I:    return GL_RGB32I;
        case Texture::Format::RGB32UI:   return GL_RGB32UI;
        case Texture::Format::RGBA8I:    return GL_RGBA8I;
        case Texture::Format::RGBA8UI:   return GL_RGBA8UI;
        case Texture::Format::RGBA16I:   return GL_RGBA16I;
        case Texture::Format::RGBA16UI:  return GL_RGBA16UI;
        case Texture::Format::RGBA32I:   return GL_RGBA32I;
        case Texture::Format::RGBA32UI:  return GL_RGBA32UI;

        default: CORE_ASSERT(false, "OpenGLTexture2D::_OpenGLFormat: Unkown format!"); return 0;
    }
}

void OpenGLTexture::Bind(unsigned int slot)
{
    m_slot = slot;
    glBindTextureUnit(slot, m_id);
//     INFO("OpenGLTexture({}-{})::Bind: {}", m_name, m_id, m_slot);
}

void OpenGLTexture::Unbind() const
{
    glBindTextureUnit(m_slot, 0);
}

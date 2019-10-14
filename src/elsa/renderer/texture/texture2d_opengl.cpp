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
    glCreateTextures(GL_TEXTURE_2D, 1, &m_id);
}

OpenGLTexture2D::~OpenGLTexture2D()
{
    glDeleteTextures(1, &m_id);
}

void OpenGLTexture2D::_LoadImage()
{
    stbi_set_flip_vertically_on_load(true);
    stbi_uc* data = stbi_load(m_imagePath.c_str(), &m_width, &m_height, &m_channel, 0);
    CORE_ASSERT(data, "OpenGLTexture2D::LoadFromFile: Failed to load image: "+m_imagePath);
    INFO("OpenGLTexture2D::_LoadImage: {}, {}x{}x{}", m_imagePath, m_width, m_height, m_channel);

    glTextureStorage2D(m_id, 1, GL_RGB8, m_width, m_height);
    glTextureParameteri(m_id, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTextureParameteri(m_id, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTextureSubImage2D(m_id, 0, 0, 0, m_width, m_height, m_channel == 3? GL_RGB : GL_RGBA, GL_UNSIGNED_BYTE, data);

    stbi_image_free(data);
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

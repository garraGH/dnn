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
#include "core.h"

std::shared_ptr<Texture2D> OpenGLTexture2D::Create(const std::string& name)
{
    return std::make_shared<OpenGLTexture2D>(name);
}

OpenGLTexture2D::OpenGLTexture2D(const std::string& name)
    : Texture(name)
    , Texture2D(name)
{
    _Create();
    INFO("texture({}-{}) created.", m_name, m_id);
}

OpenGLTexture2D::~OpenGLTexture2D()
{
    _Destroy();
    INFO("texture({}-{}-{}x{}x{}) destroyed.", m_name, m_id, m_width, m_height, m_channel);
}

void OpenGLTexture2D::_Load(bool bVerticalFlip)
{
    INFO("OpenGLTexture2D::_Load: path({}), levels({})", m_imagePath, m_levels);
        
    stbi_set_flip_vertically_on_load(bVerticalFlip);
    bool isHDR = stbi_is_hdr(m_imagePath.c_str());
    void* data = isHDR? 
        (void*)stbi_loadf(m_imagePath.c_str(), (int*)&m_width, (int*)&m_height, (int*)&m_channel, 0):
        (void*)stbi_load (m_imagePath.c_str(), (int*)&m_width, (int*)&m_height, (int*)&m_channel, 0);
    
    CORE_ASSERT(data, "OpenGLTexture2D::Load: Failed to load image: "+m_imagePath);

    GLenum baseFormat = GL_RGB;
    GLenum sizedFormat = GL_RGB8;
    GLenum dataType = GL_UNSIGNED_BYTE;
    if(isHDR)
    {
        sizedFormat = GL_RGB16;
        dataType = GL_FLOAT;
    }
    else
    {
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
    }
    glBindTexture(GL_TEXTURE_2D, m_id);
    glTextureParameteri(m_id, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTextureParameteri(m_id, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    if(m_levels == 0)
    {
        m_levels = std::log2(std::max(m_width, m_height));
    }
    glTextureStorage2D(m_id, m_levels, sizedFormat, m_width, m_height);
    glTextureSubImage2D(m_id, 0, 0, 0, m_width, m_height, baseFormat, dataType, data);

    if(m_levels>1)
    {
        GenerateMipmap();
    }

    stbi_image_free(data);

    INFO("OpenGLTexture2D::_Load: {}, {}x{}x{}, {}", m_imagePath, m_width, m_height, m_channel, m_levels);
}

void OpenGLTexture2D::_Allocate()
{
    INFO("OpenGLTexture2D::_Allocate: size({}x{}), levles({}), samples({})", m_width, m_height, m_levels, m_samples);
    if(m_samples == 1)
    {
        glBindTexture(GL_TEXTURE_2D, m_id);
        glTextureStorage2D(m_id, m_levels, _FormatOfGraphicsAPI(), m_width, m_height);
        if(m_levels>1)
        {
            GenerateMipmap();
        }
        glBindTexture(GL_TEXTURE_2D, 0);
    }
    else
    {
        glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, m_id);
        glTextureStorage2DMultisample(m_id, m_samples, _FormatOfGraphicsAPI(), m_width, m_height, GL_TRUE);
        glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, 0);
    }
    glTextureParameteri(m_id, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTextureParameteri(m_id, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
}


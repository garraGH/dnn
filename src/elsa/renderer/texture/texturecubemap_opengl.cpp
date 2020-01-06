/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : texturecubemap_opengl.cpp
* author      : Garra
* time        : 2019-12-12 13:42:34
* description : 
*
============================================*/


#include "texturecubemap_opengl.h"
#include "stb_image.h"
#include "glad/gl.h"
#include "core.h"

std::shared_ptr<TextureCubemap> OpenGLTextureCubemap::Create(const std::string& name)
{
    return std::make_shared<OpenGLTextureCubemap>(name);
}

OpenGLTextureCubemap::OpenGLTextureCubemap(const std::string& name)
    : Texture(name)
    , TextureCubemap(name)
{
    _Create();
    INFO("Create OpenGLTextureCubemap: {}", m_id);
}

OpenGLTextureCubemap::~OpenGLTextureCubemap()
{
    _Destroy();
    INFO("Destroy OpenGLTextureCubemap: {}", m_id);
}

std::string OpenGLTextureCubemap::_FaceFile(int i)
{
    if(i == 0)
    {
        return m_imagePath;
    }

    std::string filepath = m_imagePath;
    std::string::size_type pos = filepath.rfind("right");
    int n = 5;
    if(pos == std::string::npos)
    {
        pos = filepath.rfind("rt");
        n = 2;
    }

    CORE_ASSERT(pos != std::string::npos, "filename does not feed the format.");
    switch(i)
    {
        case 1: return filepath.replace(pos, n, n == 5? "left"  : "lf");
        case 2: return filepath.replace(pos, n, n == 5? "up"    : "up");
        case 3: return filepath.replace(pos, n, n == 5? "down"  : "dn");
        case 4: return filepath.replace(pos, n, n == 5? "front" : "ft");
        case 5: return filepath.replace(pos, n, n == 5? "back"  : "bk");
        default: return filepath;
    }
}

void OpenGLTextureCubemap::_Load(bool bVerticalFlip)
{
    INFO("OpenGLTexture2D::_Load: path({}), levels({})", m_imagePath, m_levels);
    glBindTexture(GL_TEXTURE_CUBE_MAP, m_id);
    stbi_set_flip_vertically_on_load(bVerticalFlip);
    for(int i=0; i<6; i++)
    {
        std::string filepath = _FaceFile(i);
        INFO(filepath);
        unsigned char* data = stbi_load(filepath.c_str(), (int*)&m_width, (int*)&m_height, (int*)&m_channel, 0);
        if(!data)
        {
            WARN("OpenGLTextureCubemap::_Load: failed at path: {}", filepath);
        }
        else
        {
            INFO("OpenGLTextureCubemap::_Load: {}, {}x{}x{}", filepath, m_width, m_height, m_channel);
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X+i, 0, GL_RGB, m_width, m_height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
        }
        stbi_image_free(data);
    }

    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_CUBE_MAP, 0);

    if(m_levels>1)
    {
        GenerateMipmap();
    }
}

void OpenGLTextureCubemap::_Allocate()
{
    glBindTexture(GL_TEXTURE_CUBE_MAP, m_id);
    glTextureStorage2D(m_id, m_levels, _FormatOfGraphicsAPI(), m_width, m_height);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_CUBE_MAP, 0);

    if(m_levels>1)
    {
        GenerateMipmap();
    }
}


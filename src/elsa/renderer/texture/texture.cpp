/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : texture.cpp
* author      : Garra
* time        : 2019-10-14 21:41:51
* description : 
*
============================================*/


#include "texture.h"
#include "logger.h"

std::shared_ptr<Texture> Texture::Load(const std::string& imagePath, bool bVerticalFlip, unsigned int levels)
{
    m_imagePath = imagePath;
    m_levels = levels;
    
    INFO("Texture::Load: path({}), levels({})", m_imagePath, m_levels);
    _Load(bVerticalFlip);

    return shared_from_this();
}

void Texture::Reload(const std::string& imagePath, bool bVerticalFlip, unsigned int levels)
{
    _Recreate();
    Load(imagePath, bVerticalFlip, levels);
}

std::shared_ptr<Texture> Texture::Set(unsigned int width, unsigned int height, Format format, unsigned int samples, unsigned int levels)
{
    if(width==m_width && height==m_height && format==m_format && samples==m_samples && levels==m_levels)
    {
        return shared_from_this();
    }

    m_width = width;
    m_height = height;
    m_format = format;
    m_samples = samples;
    m_levels = levels==0? std::log2(std::max(m_width, m_height)) : levels;
    _Allocate();

    return shared_from_this();
}

void Texture::Reset(unsigned int width, unsigned int height, Format format, unsigned int samples)
{
    _Recreate();
    Set(width, height, format, samples);
}

void Texture::_Recreate()
{
    _Destroy();
    _Create();
}


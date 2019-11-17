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

std::shared_ptr<Texture> Texture::LoadFromFile(const std::string& imagePath)
{
    m_imagePath = imagePath;
    return _LoadImage();
}


std::shared_ptr<Texture> Texture::Set(unsigned int width, unsigned int height, unsigned int samples, Format format)
{
    Reset(width, height, samples, format);
    return shared_from_this();
}

void Texture::Reset(unsigned int width, unsigned int height, unsigned int samples, Format format)
{
    if(width==m_width && height==m_height && samples==m_samples && format==m_format)
    {
        return;
    }

    m_width = width;
    m_height = height;
    m_samples = samples;
    m_format = format;

    _AllocateStorage();
}

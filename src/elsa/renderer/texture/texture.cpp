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




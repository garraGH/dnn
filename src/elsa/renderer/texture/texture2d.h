/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : texture2d.h
* author      : Garra
* time        : 2019-10-14 17:22:20
* description : 
*
============================================*/


#pragma once
#include "texture.h"

class Texture2D : public Texture, public std::enable_shared_from_this<Texture2D>
{
public:
    Texture2D(const std::string& name) : Texture(name) {} 

    virtual std::string GetTypeName() const { return "Texture2D"; }
    std::shared_ptr<Texture2D> LoadFromFile(const std::string& imagePath);
    const std::string& GetImagePath() const { return m_imagePath; }


    int GetWidth() const { return m_width; }
    int GetHeight() const { return m_height; }


    static std::shared_ptr<Texture2D> Create(const std::string& name);

protected:
    virtual void _LoadImage() {}

protected:
    std::string m_imagePath;
    int m_width = 0;
    int m_height = 0;
    int m_channel = 0;
};

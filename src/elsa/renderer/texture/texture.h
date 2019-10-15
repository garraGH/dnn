/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : /home/garra/study/dnn/src/elsa/renderer/texture/texture.h
* author      : Garra
* time        : 2019-10-14 21:42:04
* description : 
*
============================================*/


#pragma once
#include "../rendererobject.h"

class Texture : public RenderObject, public std::enable_shared_from_this<Texture>
{
public:
    Texture(const std::string& name) : RenderObject(name) {}
    virtual std::string GetTypeName() { return "Texture"; }

    std::shared_ptr<Texture> LoadFromFile(const std::string& imagePath);
    const std::string& GetImagePath() const { return m_imagePath; }
    int GetWidth() const { return m_width; }
    int GetHeight() const { return m_height; }
    int GetDepth() const { return m_depth; }
    int GetChannel() const { return m_channel; }

protected:
    virtual std::shared_ptr<Texture> _LoadImage() { return nullptr; }

protected:
    std::string m_imagePath;
    int m_width = 0;
    int m_height = 0;
    int m_depth = 0;
    int m_channel = 0;
};

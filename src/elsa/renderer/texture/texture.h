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
    enum class Format
    {
        R8, R8SN, R16, R16SN, RG8, RG8SN, RG16, RG16SN, RG3B2, RGB4, RGB5, RGB8, RGB8SN, RGB10, RGB12, RGB16SN, RGBA2, RGBA4, RGB5A1, RGBA8, RGBA8SN, RGB10A2, RGB10A2UI, RGBA12, RGBA16, SRGB8, SRGBA8, R16F, RG16F, RGB16F, RGBA16F, R32F, RG32F, RGB32F, RGBA32F, RG11B10F, RGB9E5, R8I, R8UI, R16I, R16UI, R32I, R32UI, RG8I, RG8UI, RG16I, RG16UI, RG32I, RG32UI, RGB8I, RGB8UI, RGB16I, RGB16UI, RGB32I, RGB32UI, RGBA8I, RGBA8UI, RGBA16I, RGBA16UI, RGBA32I, RGBA32UI
    };

    Texture(const std::string& name) : RenderObject(name) {}
    static std::string GetTypeName() { return "Texture"; }

    void SetImagePath(const std::string& imagepath) { m_imagePath = imagepath; }
    std::shared_ptr<Texture> Load(const std::string& imagePath, bool bVerticalFlip=true, unsigned int levels=0);
    void Reload(const std::string& imagePath, bool bVerticalFlip=true, unsigned int levels=0);
    std::shared_ptr<Texture> Set(unsigned int width, unsigned int height, Format format=Format::RGB8, unsigned int samples=1, unsigned int levels=1);
    void Reset(unsigned int width, unsigned int height, Format format=Format::RGB8, unsigned int samples=1);
    const std::string& GetImagePath() const { return m_imagePath; }
    unsigned int GetWidth() const { return m_width; }
    unsigned int GetHeight() const { return m_height; }
    unsigned int GetDepth() const { return m_depth; }
    unsigned int GetChannel() const { return m_channel; }
    Format GetFormat() const { return m_format; }

    virtual void GenerateMipmap() = 0;


protected:
    virtual void _Load(bool bVerticalFlip) = 0;
    virtual void _Allocate() = 0;
    virtual void _Create() = 0;
    virtual void _Destroy() = 0;
    virtual unsigned int _FormatOfGraphicsAPI() = 0;

private:
    void _Recreate();

protected:
    std::string m_imagePath;
    unsigned int m_width = 0;
    unsigned int m_height = 0;
    unsigned int m_depth = 0;
    unsigned int m_channel = 0;
    unsigned int m_levels = 1;
    Format m_format = Format::RGB8;
    unsigned int m_samples = 1;
};

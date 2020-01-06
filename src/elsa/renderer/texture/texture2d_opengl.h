/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : texture2d_opengl.h
* author      : Garra
* time        : 2019-10-14 20:09:05
* description : 
*
============================================*/


#pragma once
#include "texture2d.h"
#include "texture_opengl.h"

class OpenGLTexture2D : public Texture2D, public OpenGLTexture
{
public:
    OpenGLTexture2D(const std::string& name);
    ~OpenGLTexture2D();

    static std::shared_ptr<Texture2D> Create(const std::string& name);

protected:
    virtual void _Load(bool bVerticalFlip) override;
    virtual void _Allocate() override;
};

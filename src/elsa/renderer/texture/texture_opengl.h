/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : /home/garra/study/dnn/src/elsa/renderer/texture/texture_opengl.h
* author      : Garra
* time        : 2019-12-14 13:49:28
* description : 
*
============================================*/


#pragma once

#include "texture.h"

class OpenGLTexture : virtual public Texture
{
public:
    virtual void GenerateMipmap() override;
    virtual void Bind(unsigned int slot) override;
    virtual void Unbind() const override;
    
protected:
    virtual void _Create() override ;
    virtual void _Destroy() override;
    virtual unsigned int _FormatOfGraphicsAPI() override;

protected:
    unsigned int m_slot = 0;
};

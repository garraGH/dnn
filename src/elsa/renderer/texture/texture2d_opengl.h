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

class OpenGLTexture2D : public Texture2D
{
public:
    OpenGLTexture2D(const std::string& name);
    ~OpenGLTexture2D();


    virtual void Bind(unsigned int slot = 0) override;
    virtual void Unbind() const override;

    static std::shared_ptr<Texture2D> Create(const std::string& name);

protected:
    virtual void _LoadImage() override;

private:
    unsigned int m_slot = 0;
};

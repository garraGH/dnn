/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : /home/garra/study/dnn/src/elsa/renderer/texture/texturecubemap_opengl.h
* author      : Garra
* time        : 2019-12-12 13:42:35
* description : 
*
============================================*/


#pragma once
#include "texturecubemap.h"
#include "texture_opengl.h"

class OpenGLTextureCubemap : public TextureCubemap, public OpenGLTexture
{
public:
    OpenGLTextureCubemap(const std::string& name);
    ~OpenGLTextureCubemap();

    static std::shared_ptr<TextureCubemap> Create(const std::string& name);

protected:
   virtual void _Load(bool bVerticalFlip) override;
   virtual void _Allocate() override;

private:
   std::string _FaceFile(int i);
};

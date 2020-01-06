/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : texturecubemap.cpp
* author      : Garra
* time        : 2019-12-12 13:34:46
* description : 
*
============================================*/


#include "../renderer.h"
#include "texturecubemap.h"
#include "texturecubemap_opengl.h"

std::shared_ptr<TextureCubemap> TextureCubemap::Create(const std::string& name)
{
    switch(Renderer::GetAPIType())
    {
        case Renderer::API::OpenGL: return OpenGLTextureCubemap::Create(name);
        default: return nullptr;
    }
}




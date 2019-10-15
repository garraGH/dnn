/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : texture2d.cpp
* author      : Garra
* time        : 2019-10-14 17:22:20
* description : 
*
============================================*/

#include "../renderer.h"
#include "texture2d.h"
#include "texture2d_opengl.h"

std::shared_ptr<Texture2D> Texture2D::Create(const std::string& name)
{
    switch(Renderer::GetAPIType())
    {
        case Renderer::API::OpenGL: return OpenGLTexture2D::Create(name);
        default: return nullptr;
    }
}

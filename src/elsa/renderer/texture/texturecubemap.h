/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : /home/garra/study/dnn/src/elsa/renderer/texture/texturecubemap.h
* author      : Garra
* time        : 2019-12-12 13:34:56
* description : 
*
============================================*/


#pragma once
#include "texture.h"

class TextureCubemap : virtual public Texture
{
public:
    enum class Face : int
    {
        POSITIVE_X = 0, 
        NEGATIVE_X, 
        POSITIVE_Y, 
        NEGATIVE_Y, 
        POSITIVE_Z, 
        NEGATIVE_Z,
    };

    TextureCubemap(const std::string& name) : Texture(name){}

    static std::string GetTypeName() { return "TextureCubemap"; }
    static std::shared_ptr<TextureCubemap> Create(const std::string& name);
};

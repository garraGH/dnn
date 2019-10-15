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

class Texture2D : public Texture
{
public:
    Texture2D(const std::string& name) : Texture(name) {} 

    virtual std::string GetTypeName() const { return "Texture2D"; }
    static std::shared_ptr<Texture2D> Create(const std::string& name);
};

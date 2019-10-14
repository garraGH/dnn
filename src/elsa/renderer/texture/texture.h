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

class Texture : public RenderObject
{
public:
    Texture(const std::string& name) : RenderObject(name) {}
};

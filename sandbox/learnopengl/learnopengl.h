/*============================================
* Copyright(C)2020 Garra. All rights reserved.
* 
* file        : sandbox/learnopengl/learnopengl.h
* author      : Garra
* time        : 2020-01-06 20:35:13
* description : 
*
============================================*/


#pragma once
#include "elsa.h"
class LearnOpenGL
{
public:
    virtual void OnUpdate() = 0;
    virtual void OnEvent(Event& e) = 0;
    virtual void OnImgui() = 0;
};

/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : context.h
* author      : Garra
* time        : 2019-09-30 16:51:10
* description : 
*
============================================*/


#pragma once

class GraphicsContext
{
public:
    virtual void Init() = 0;
    virtual void SwapBuffers() = 0;
};

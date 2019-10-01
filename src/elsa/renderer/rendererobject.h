/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : rendererobject.h
* author      : Garra
* time        : 2019-10-01 21:57:24
* description : 
*
============================================*/


#pragma once

#define RenderObjectID unsigned int 

class RenderObject
{
public:
    virtual ~RenderObject() {}
    virtual void Bind(unsigned int slot=0) const = 0;
    virtual void Unbind() const = 0;
protected:
    RenderObjectID m_id = 0;
};

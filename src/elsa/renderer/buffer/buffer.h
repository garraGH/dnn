/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : buffer.h
* author      : Garra
* time        : 2019-10-01 22:13:27
* description : 
*
============================================*/


#pragma once
#include "../rendererobject.h"

class Buffer : public RenderObject
{
public:
    static Buffer* CreateVertex(unsigned int size, float* data);
    static Buffer* CreateIndex(unsigned int size, unsigned int* data);

public:
    unsigned int GetCount() const;

protected:
    unsigned int m_count = 0;
};

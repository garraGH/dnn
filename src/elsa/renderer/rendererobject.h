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
#include <string>
#include <memory>

#define RenderObjectID unsigned int 

class RenderObject
{
public:
    RenderObject(const std::string& name="unnamed") : m_name(name) {}
    virtual ~RenderObject() {}
    virtual void Bind(unsigned int slot=0) const = 0;
    virtual void Unbind() const = 0;

    RenderObjectID ID() const { return m_id; }
    const std::string& GetName() const { return m_name; }
protected:
    RenderObjectID m_id = 0;
    std::string m_name;
};

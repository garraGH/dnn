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

class Asset
{
public:
    Asset(const std::string& name = "unnamed") : m_name(name) {}
    const std::string& GetName() const { return m_name; }
    static std::shared_ptr<Asset> Create(const std::string& name) { return std::make_shared<Asset>(name); }
    virtual std::string GetTypeName() const { return "Asset"; }

protected:
    std::string m_name;
};

class RenderObject : public Asset
{
public:
    RenderObject(const std::string& name="unnamed") : Asset(name) {}
    virtual ~RenderObject() {}
    virtual void Bind(unsigned int slot=0) const = 0;
    virtual void Unbind() const = 0;
    virtual std::string GetTypeName() const { return "RenderObject"; }
    RenderObjectID ID() const { return m_id; }

protected:
    RenderObjectID m_id = 0;
};

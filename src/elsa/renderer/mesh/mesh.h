/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : mesh.h
* author      : Garra
* time        : 2019-10-05 23:47:24
* description : 
*
============================================*/


#pragma once

#include <memory>
#include <vector>
#include "../shader/shader.h"
#include "../transform/transform.h"
#include "../buffer/buffer.h"
#include "../rendererobject.h"

class Mesh : public Asset, public std::enable_shared_from_this<Mesh>
{
public:
    Mesh(const std::string& name="unnamed") : Asset(name) {} 
    virtual void Bind(const std::shared_ptr<Shader>& shader) = 0;
    std::shared_ptr<Mesh> Set(const std::shared_ptr<Buffer>& ib, const std::vector< std::shared_ptr<Buffer> >& vbs, const std::shared_ptr<Transform>& trans=std::make_shared<Transform>())
    {
        m_bufferArray->SetIndexBuffer(ib);
        for(const auto& vb : vbs)
        {
            m_bufferArray->AddVertexBuffer(vb);
        }
        m_transform = trans;
        m_dirty = true;
        return shared_from_this();
    }

    std::shared_ptr<Mesh> SetTransform(const std::shared_ptr<Transform>& trans) { m_transform = trans; m_dirty = true; return shared_from_this(); }
    std::shared_ptr<Mesh> SetIndexBuffer(const std::shared_ptr<Buffer>& indexBuffer) { m_bufferArray->SetIndexBuffer(indexBuffer); m_dirty = true; return shared_from_this(); }
    std::shared_ptr<Mesh> AddVertexBuffer(const std::shared_ptr<Buffer>& vertexBuffer) { m_bufferArray->AddVertexBuffer(vertexBuffer); m_dirty = true; return shared_from_this(); }

    const std::shared_ptr<BufferArray>& GetBufferArray() const { return m_bufferArray; }
    static std::shared_ptr<Mesh> Create(const std::string& name);

protected:
    bool m_dirty = true;
    std::shared_ptr<Shader> m_shader = nullptr;
    std::shared_ptr<BufferArray> m_bufferArray = BufferArray::Create();
    std::shared_ptr<Transform> m_transform = std::make_shared<Transform>();
};

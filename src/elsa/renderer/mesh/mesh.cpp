/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : mesh.cpp
* author      : Garra
* time        : 2019-10-05 23:47:24
* description : 
*
============================================*/


#include "mesh.h"

std::shared_ptr<Mesh> Mesh::Set(const std::shared_ptr<Buffer>& ib, const std::vector< std::shared_ptr<Buffer> >& vbs, const std::shared_ptr<Transform>& trans)
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
std::shared_ptr<Mesh> Mesh::SetTransform(const std::shared_ptr<Transform>& trans) 
{ 
    m_transform = trans;
    m_dirty = true;
    return shared_from_this();
}

std::shared_ptr<Mesh> Mesh::SetIndexBuffer(const std::shared_ptr<Buffer>& indexBuffer)
{ 
    m_bufferArray->SetIndexBuffer(indexBuffer); 
    m_dirty = true;
    return shared_from_this();
}

std::shared_ptr<Mesh> Mesh::AddVertexBuffer(const std::shared_ptr<Buffer>& vertexBuffer)
{
    m_bufferArray->AddVertexBuffer(vertexBuffer);
    m_dirty = true;
    return shared_from_this();
}

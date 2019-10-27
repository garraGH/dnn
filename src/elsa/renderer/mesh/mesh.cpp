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
namespace Elsa {

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

void Mesh::SetVertexNumber(unsigned int numVertices)
{
    m_vertices.reserve(numVertices);
    INFO("Mesh::SetVertexNumber: {}", m_vertices.size());
}


void Mesh::PushVertex(const glm::vec3& vtx)
{
    m_vertices.push_back(vtx);
}

void Mesh::SetIndexNumber(unsigned int nIndeices)
{
    m_indices.reserve(nIndeices);
    INFO("Mesh::SetIndexNumber: {}", m_indices.size());
}

void Mesh::PushIndex(unsigned int index)
{
    m_indices.push_back(index);
}

void Mesh::Build()
{
    Buffer::Layout layoutVextex = { {Buffer::Element::DataType::Float3, "a_Position", false}, };
    INFO("Mesh::Build: {} numVertices: {}, numIndices: {}", m_name, m_vertices.size(), m_indices.size());
    std::shared_ptr<Buffer> vb = Buffer::CreateVertex(m_vertices.size()*sizeof(glm::vec3), &m_vertices[0]);
    vb->SetLayout(layoutVextex);

    std::shared_ptr<Buffer> ib = Buffer::CreateIndex(m_indices.size()*sizeof(unsigned int), &m_indices[0]);
    Buffer::Layout layoutIndex = { {Buffer::Element::DataType::UInt} };
    ib->SetLayout(layoutIndex);

    SetIndexBuffer(ib);
    AddVertexBuffer(vb);

}

}


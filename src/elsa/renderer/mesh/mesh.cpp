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

std::shared_ptr<Mesh> Mesh::Set(const std::shared_ptr<Buffer>& ib, const std::vector< std::shared_ptr<Buffer> >& vbs)
{
    m_bufferArray->Clear();
    m_bufferArray->SetIndexBuffer(ib);
    for(const auto& vb : vbs)
    {
        m_bufferArray->AddVertexBuffer(vb);
    }
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
}


void Mesh::PushVertex(const Vertex& vtx)
{
    m_vertices.push_back(vtx);
}

void Mesh::SetIndexNumber(unsigned int nIndeices)
{
    m_indices.reserve(nIndeices);
}

void Mesh::PushIndex(unsigned int index)
{
    m_indices.push_back(index);
}

void Mesh::SetAABB(float x0, float y0, float z0, float x1, float y1, float z1)
{
    m_min = glm::vec3(x0, y0, z0);
    m_max = glm::vec3(x1, y1, z1);
}

std::pair<glm::vec3, glm::vec3> Mesh::GetAABB() const
{
    return {m_min, m_max};
}

void Mesh::Build()
{
    Buffer::Layout layoutVextex = { 
        { Buffer::Element::DataType::Float3, "a_Position", false }, 
        { Buffer::Element::DataType::Float3, "a_Normal", false }, 
        { Buffer::Element::DataType::Float2, "a_TexCoord", false }
    };

    INFO("Mesh::Build: {} numVertices: {}, numIndices: {}", m_name, m_vertices.size(), m_indices.size());
    std::shared_ptr<Buffer> vb = Buffer::CreateVertex(m_vertices.size()*sizeof(Vertex), &m_vertices[0]);
    vb->SetLayout(layoutVextex);

    std::shared_ptr<Buffer> ib = Buffer::CreateIndex(m_indices.size()*sizeof(unsigned int), &m_indices[0]);
    Buffer::Layout layoutIndex = { {Buffer::Element::DataType::UInt} };
    ib->SetLayout(layoutIndex);

    SetIndexBuffer(ib);
    AddVertexBuffer(vb);

}

}


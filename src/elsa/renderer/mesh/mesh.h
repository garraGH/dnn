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
#include "../material/material.h"

namespace Elsa {

class Mesh : public Asset, public std::enable_shared_from_this<Mesh>
{
public:
//     class Vertexes
//     {
//     public:
//         enum class DataLayout
//         {
// 
//         };
// 
//         class Vertex
//         {
//         public:
//         protected:
//         private:
//             glm::vec3 m_pos;        // position
//             glm::vec3 m_nor;        // normal
//             glm::vec2 m_uv;         // texcoordinate
//         };
//     public:
//         Vertexes();
//     protected:
//     private:
//         int m_count;
//         void* m_data;
//         DataLayout m_dataLayout;
//     };
// 
//     class Indices
//     {
//         enum class DataType
//         {
//             Int8, 
//             Int16, 
//             Int32, 
//         };
// 
//     public:
//     protected:
//     private:
//         int m_count;
//         void* m_data;
//         DataType m_dataType;
//     };
    struct Vertex
    {
        glm::vec3 pos;
        glm::vec3 nor;
        glm::vec2 uv;
    };

public:
    Mesh(const std::string& name="unnamed") : Asset(name) {} 
    virtual void Bind(const std::shared_ptr<Shader>& shader) = 0;
    virtual std::string GetTypeName() const override { return "Mesh"; }

    std::shared_ptr<Mesh> Set(const std::shared_ptr<Buffer>& ib, const std::vector< std::shared_ptr<Buffer> >& vbs, const std::shared_ptr<Transform>& trans=std::make_shared<Transform>());
    std::shared_ptr<Mesh> SetTransform(const std::shared_ptr<Transform>& trans);
    std::shared_ptr<Mesh> SetIndexBuffer(const std::shared_ptr<Buffer>& indexBuffer);
    std::shared_ptr<Mesh> AddVertexBuffer(const std::shared_ptr<Buffer>& vertexBuffer);

    const std::shared_ptr<BufferArray>& GetBufferArray() const { return m_bufferArray; }
    static std::shared_ptr<Mesh> Create(const std::string& name);

    void SetVertexNumber(unsigned int nVertexes);
    void SetIndexNumber(unsigned int nIndeices);
    void PushVertex(const Vertex& vtx);
    void PushIndex(unsigned int index);
    void SetAABB(float x0, float y0, float z0, float x1, float y1, float z1);
    std::pair<glm::vec3, glm::vec3> GetAABB() const;

    void Build();

    const std::vector<Vertex>& GetVertices() const { return m_vertices; }
protected:
    bool m_dirty = true;
    std::shared_ptr<Shader> m_shader = nullptr;
    std::shared_ptr<BufferArray> m_bufferArray = BufferArray::Create();
    std::shared_ptr<Transform> m_transform = std::make_shared<Transform>();
    std::shared_ptr<Material> m_material = nullptr;

    std::vector<Vertex> m_vertices;
    std::vector<unsigned int> m_indices;
    glm::vec3 m_min = glm::vec3(0);
    glm::vec3 m_max = glm::vec3(1);
};

}

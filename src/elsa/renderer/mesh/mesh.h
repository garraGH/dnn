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
#include "../shader/shader.h"
#include "../transform/transform.h"
#include "../buffer/buffer.h"

class Mesh
{
public:
    Mesh(const std::string& name) : m_name(name) {} 
    virtual void Bind(const std::shared_ptr<Shader>& shader) = 0;

    void SetTransform(const std::shared_ptr<Transform>& transform) { m_transform = transform; m_dirty = true; }
    void SetIndexBuffer(const std::shared_ptr<Buffer>& indexBuffer) { m_bufferArray->SetIndexBuffer(indexBuffer); m_dirty = true; }
    void AddVertexBuffer(const std::shared_ptr<Buffer>& vertexBuffer) { m_bufferArray->AddVertexBuffer(vertexBuffer); m_dirty = true; }

    const std::shared_ptr<BufferArray>& GetBufferArray() const { return m_bufferArray; }

    static Mesh* Create(const std::string& anme);

protected:
    bool m_dirty = true;
    std::shared_ptr<BufferArray> m_bufferArray = std::shared_ptr<BufferArray>(BufferArray::Create());
    std::shared_ptr<Shader> m_shader = nullptr;
    std::shared_ptr<Transform> m_transform = std::make_shared<Transform>();

private:
    const std::string m_name = "undefined";
};

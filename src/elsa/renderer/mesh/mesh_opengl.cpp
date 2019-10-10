/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : mesh_opengl.cpp
* author      : Garra
* time        : 2019-10-06 00:34:20
* description : 
*
============================================*/


#include "mesh_opengl.h"

std::shared_ptr<Mesh> Mesh::Create(const std::string& name)
{
    return std::make_shared<OpenGLMesh>(name);
}

void OpenGLMesh::Bind(const std::shared_ptr<Shader>& shader)
{
    m_bufferArray->Bind(shader);
    if(!shader || (m_shader == shader && !m_dirty))
    {
        return;
    }
    m_dirty = false;
    m_shader = shader;

    m_shader->SetTransformMatrix(m_transform->GetTransformMatrx());
}

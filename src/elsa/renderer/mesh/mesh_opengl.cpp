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
namespace Elsa {

std::shared_ptr<Mesh> Mesh::Create(const std::string& name)
{
    return std::make_shared<OpenGLMesh>(name);
}

void OpenGLMesh::Bind(const std::shared_ptr<Shader>& shader)
{
    CORE_ASSERT(m_bufferArray, "null bufferArray!");
//     INFO("OpenGLMesh::Bind: mesh {}, shader {}", m_name, shader->GetName());
    m_bufferArray->Bind(shader);
//     if(!shader || (m_shader == shader && !m_dirty))
//     {
//         return;
//     }
//     INFO("OpenGLMesh::Bind: ShaderChanged");
    m_dirty = false;
    m_shader = shader;

}

}

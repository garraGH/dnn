/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : material_opengl.cpp
* author      : Garra
* time        : 2019-10-05 22:48:49
* description : 
*
============================================*/


#include "material_opengl.h"
#include "../../core.h"
#include "glad/gl.h"

std::shared_ptr<Material> OpenGLMaterial::Create(const std::string& name)
{
    return std::make_shared<OpenGLMaterial>(name);
}

void OpenGLMaterial::Bind(const std::shared_ptr<Shader>& shader)
{
//     if( !shader || (shader == m_shader && !m_dirty))
//     {
//         return;
//     }

//     INFO("OpenGLMaterial::Bind: ShaderChanged");

    m_shader = shader;
    m_dirty = false;

    _BindUniforms(shader);
    _BindTextures(shader);
    _BindUniformBuffers(shader);
}

void OpenGLMaterial::_BindUniforms(const std::shared_ptr<Shader>& shader)
{
    using MUT = Material::Uniform::Type;
    for(auto& u : m_uniforms)
    {
        int location = m_shader->GetUniformLocation(u.first);
//         INFO("OpenGLMaterial::Bind: {}, {}", u.first, location);
        if(location == -1)
        {
            continue;
        }

        int count = u.second->GetCount();
        const void* data = u.second->GetData();
        bool transpose = u.second->NeedTranspose();

        switch(u.second->GetType())
        {
            case MUT::Float1: glUniform1fv (location, count, (GLfloat*)data); break; 
            case MUT::Float2: glUniform2fv (location, count, (GLfloat*)data); break;
            case MUT::Float3: glUniform3fv (location, count, (GLfloat*)data); break;
            case MUT::Float4: glUniform4fv (location, count, (GLfloat*)data); break;
            case MUT::Int1:   glUniform1iv (location, count, (GLint*  )data); break;
            case MUT::Int2:   glUniform2iv (location, count, (GLint*  )data); break;
            case MUT::Int3:   glUniform3iv (location, count, (GLint*  )data); break;
            case MUT::Int4:   glUniform4iv (location, count, (GLint*  )data); break;
            case MUT::UInt1:  glUniform1uiv(location, count, (GLuint* )data); break;
            case MUT::UInt2:  glUniform2uiv(location, count, (GLuint* )data); break;
            case MUT::UInt3:  glUniform3uiv(location, count, (GLuint* )data); break;
            case MUT::UInt4:  glUniform4uiv(location, count, (GLuint* )data); break;
            case MUT::Mat2x2: glUniformMatrix2fv  (location, count, transpose, (GLfloat*)data); break;
            case MUT::Mat2x3: glUniformMatrix2x3fv(location, count, transpose, (GLfloat*)data); break;
            case MUT::Mat2x4: glUniformMatrix2x4fv(location, count, transpose, (GLfloat*)data); break;
            case MUT::Mat3x2: glUniformMatrix3x2fv(location, count, transpose, (GLfloat*)data); break;
            case MUT::Mat3x3: glUniformMatrix3fv  (location, count, transpose, (GLfloat*)data); break;
            case MUT::Mat3x4: glUniformMatrix3x4fv(location, count, transpose, (GLfloat*)data); break;
            case MUT::Mat4x2: glUniformMatrix4x2fv(location, count, transpose, (GLfloat*)data); break;
            case MUT::Mat4x3: glUniformMatrix4x3fv(location, count, transpose, (GLfloat*)data); break;
            case MUT::Mat4x4: glUniformMatrix4fv  (location, count, transpose, (GLfloat*)data); break;
                                                                                              
            default: CORE_ASSERT(false, "OpenGLMaterial::_BindAttribute: Unknown MaterialAttributeType!");                     
        }                                                                                     
    }
}

void OpenGLMaterial::_BindTextures(const std::shared_ptr<Shader>& shader)
{
    int slot = 0;
    for(auto& tex : m_textures)
    {
        int location = m_shader->GetUniformLocation(tex.first);
        if(location == -1)
        {
            continue;
        }

        tex.second->Bind(slot);
        glUniform1i(location, slot++);
    }
}

void OpenGLMaterial::_BindUniformBuffers(const std::shared_ptr<Shader>& shader)
{
    unsigned int bindingPoint = 0;
    for(auto ub : m_uniformBuffers)
    {
        GLuint indexOfUniformBlock = m_shader->GetUniformBlockIndex(ub.first);
        if(indexOfUniformBlock == GL_INVALID_INDEX)
        {
            continue;
        }
        ub.second->Bind();
        glBindBufferBase(GL_UNIFORM_BUFFER, bindingPoint, ub.second->ID());
        ub.second->Unbind();
        glUniformBlockBinding(shader->ID(), indexOfUniformBlock, bindingPoint);
        INFO("ub: {}, indexOfUniformBlock: {}, bindingPoint: {}", ub.first, indexOfUniformBlock, bindingPoint);
        bindingPoint++;
    }
}


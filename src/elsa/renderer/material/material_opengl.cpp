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

    _BindAttribute(shader);
    _BindTexture(shader);
}

void OpenGLMaterial::_BindAttribute(const std::shared_ptr<Shader>& shader)
{
    using MAT = Material::Attribute::Type;
    for(auto& a : m_attributes)
    {
        int location = m_shader->GetLocation(a.first);
        INFO("OpenGLMaterial::Bind: {}, {}", a.first, location);
        if(location == -1)
        {
            continue;
        }

        int count = a.second->GetCount();
        const void* data = a.second->GetData();
        bool transpose = a.second->NeedTranspose();

        switch(a.second->GetType())
        {
            case MAT::Float1: glUniform1fv (location, count, (GLfloat*)data); break; 
            case MAT::Float2: glUniform2fv (location, count, (GLfloat*)data); break;
            case MAT::Float3: glUniform3fv (location, count, (GLfloat*)data); break;
            case MAT::Float4: glUniform4fv (location, count, (GLfloat*)data); break;
            case MAT::Int1:   glUniform1iv (location, count, (GLint*  )data); break;
            case MAT::Int2:   glUniform2iv (location, count, (GLint*  )data); break;
            case MAT::Int3:   glUniform3iv (location, count, (GLint*  )data); break;
            case MAT::Int4:   glUniform4iv (location, count, (GLint*  )data); break;
            case MAT::UInt1:  glUniform1uiv(location, count, (GLuint* )data); break;
            case MAT::UInt2:  glUniform2uiv(location, count, (GLuint* )data); break;
            case MAT::UInt3:  glUniform3uiv(location, count, (GLuint* )data); break;
            case MAT::UInt4:  glUniform4uiv(location, count, (GLuint* )data); break;
            case MAT::Mat2x2: glUniformMatrix2fv  (location, count, transpose, (GLfloat*)data); break;
            case MAT::Mat2x3: glUniformMatrix2x3fv(location, count, transpose, (GLfloat*)data); break;
            case MAT::Mat2x4: glUniformMatrix2x4fv(location, count, transpose, (GLfloat*)data); break;
            case MAT::Mat3x2: glUniformMatrix3x2fv(location, count, transpose, (GLfloat*)data); break;
            case MAT::Mat3x3: glUniformMatrix3fv  (location, count, transpose, (GLfloat*)data); break;
            case MAT::Mat3x4: glUniformMatrix3x4fv(location, count, transpose, (GLfloat*)data); break;
            case MAT::Mat4x2: glUniformMatrix4x2fv(location, count, transpose, (GLfloat*)data); break;
            case MAT::Mat4x3: glUniformMatrix4x3fv(location, count, transpose, (GLfloat*)data); break;
            case MAT::Mat4x4: glUniformMatrix4fv  (location, count, transpose, (GLfloat*)data); break;
                                                                                              
            default: CORE_ASSERT(false, "OpenGLMaterial::_BindAttribute: Unknown MaterialAttributeType!");                     
        }                                                                                     
    }
}

void OpenGLMaterial::_BindTexture(const std::shared_ptr<Shader>& shader)
{
    int slot = 0;
    for(auto& tex : m_textures)
    {
        int location = m_shader->GetLocation(tex.first);
        if(location == -1)
        {
            continue;
        }

        tex.second->Bind(slot);
        glUniform1i(location, slot);
        slot++;
    }
}



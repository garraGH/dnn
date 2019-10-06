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

std::shared_ptr<Material> Material::Create(const std::string& name)
{
    return std::make_shared<OpenGLMaterial>(name);
}

void OpenGLMaterial::Bind(const std::shared_ptr<Shader>& shader)
{
    if(shader == m_shader && !m_dirty)
    {
        return;
    }

    m_shader = shader;
    m_dirty = false;

    using MAT = Material::Attribute::Type;
    for(auto& a : m_attributes)
    {
        int location = m_shader->GetLocation(a.first);
        if(location == -1)
        {
            continue;
        }

        int count = a.second->GetCount();
        const void* data = a.second->GetData();
        bool transpose = a.second->NeedTranspose();

        switch(a.second->GetType())
        {
            case MAT::Float1: return glUniform1fv (location, count, (GLfloat*)data);
            case MAT::Float2: return glUniform2fv (location, count, (GLfloat*)data);
            case MAT::Float3: return glUniform3fv (location, count, (GLfloat*)data);
            case MAT::Float4: return glUniform4fv (location, count, (GLfloat*)data);
            case MAT::Int1:   return glUniform1iv (location, count, (GLint*  )data);
            case MAT::Int2:   return glUniform2iv (location, count, (GLint*  )data);
            case MAT::Int3:   return glUniform3iv (location, count, (GLint*  )data);
            case MAT::Int4:   return glUniform4iv (location, count, (GLint*  )data);
            case MAT::UInt1:  return glUniform1uiv(location, count, (GLuint* )data);
            case MAT::UInt2:  return glUniform2uiv(location, count, (GLuint* )data);
            case MAT::UInt3:  return glUniform3uiv(location, count, (GLuint* )data);
            case MAT::UInt4:  return glUniform4uiv(location, count, (GLuint* )data);
            case MAT::Mat2x2: return glUniformMatrix2fv  (location, count, transpose, (GLfloat*)data);
            case MAT::Mat2x3: return glUniformMatrix2x3fv(location, count, transpose, (GLfloat*)data);
            case MAT::Mat2x4: return glUniformMatrix2x4fv(location, count, transpose, (GLfloat*)data);
            case MAT::Mat3x2: return glUniformMatrix3x2fv(location, count, transpose, (GLfloat*)data);
            case MAT::Mat3x3: return glUniformMatrix3fv  (location, count, transpose, (GLfloat*)data);
            case MAT::Mat3x4: return glUniformMatrix3x4fv(location, count, transpose, (GLfloat*)data);
            case MAT::Mat4x2: return glUniformMatrix4x2fv(location, count, transpose, (GLfloat*)data);
            case MAT::Mat4x3: return glUniformMatrix4x3fv(location, count, transpose, (GLfloat*)data);
            case MAT::Mat4x4: return glUniformMatrix4fv  (location, count, transpose, (GLfloat*)data);

            default: CORE_ASSERT(false, "Unkown MaterialAttributeType!");
        }
    }
}

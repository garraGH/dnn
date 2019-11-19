/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : material.cpp
* author      : Garra
* time        : 2019-10-05 21:55:48
* description : 
*
============================================*/


#include "../renderer.h"
#include "material.h"
#include "material_opengl.h"

std::shared_ptr<Material> Material::Create(const std::string& name)
{
    switch(Renderer::GetAPIType())
    {
        case Renderer::API::OpenGL: return OpenGLMaterial::Create(name);
//         default: CORE_ASSERT(false, "Material::Create: API is currently not supportted!");
        default: return nullptr;
    }
}

int Material::Uniform::_TypeSize() const
{
    switch(m_type)
    {
        case Type::Float1: return 1*sizeof(float);
        case Type::Float2: return 2*sizeof(float);
        case Type::Float3: return 3*sizeof(float);
        case Type::Float4: return 4*sizeof(float);
        case Type::Int1:   return 1*sizeof(int);
        case Type::Int2:   return 2*sizeof(int);
        case Type::Int3:   return 3*sizeof(int);
        case Type::Int4:   return 4*sizeof(int);
        case Type::UInt1:  return 1*sizeof(unsigned int);
        case Type::UInt2:  return 2*sizeof(unsigned int);
        case Type::UInt3:  return 3*sizeof(unsigned int);
        case Type::UInt4:  return 4*sizeof(unsigned int);
        case Type::Mat2x2: return 2*2*sizeof(float);
        case Type::Mat2x3: return 2*3*sizeof(float);
        case Type::Mat2x4: return 2*4*sizeof(float);
        case Type::Mat3x2: return 3*2*sizeof(float);
        case Type::Mat3x3: return 3*3*sizeof(float);
        case Type::Mat3x4: return 3*4*sizeof(float);
        case Type::Mat4x2: return 4*2*sizeof(float);
        case Type::Mat4x3: return 4*3*sizeof(float);
        case Type::Mat4x4: return 4*4*sizeof(float);
        default: CORE_ASSERT(false, "Unknown UniformType!"); return 0;
    }
}

void Material::Uniform::_AllocateData()
{
    int size = m_count*_TypeSize();
    m_data = std::shared_ptr<char>(new char[size], [](char* p) { delete[] p; });
}

std::shared_ptr<Material::Uniform> Material::Uniform::Set(Type type, int cnt, const void* data, bool transpose)
{ 
    m_type = type;
    m_count = cnt;
    m_transpose = transpose;
    return SetData(data);
}

std::shared_ptr<Material::Uniform> Material::Uniform::SetType(Type type)
{
    m_type = type;
    return shared_from_this();
}

std::shared_ptr<Material::Uniform> Material::Uniform::SetData(const void* data)
{
    if(data != nullptr)
    {
        _AllocateData();
        UpdateData(data);
    }
    return shared_from_this();
}

std::shared_ptr<Material::Uniform> Material::Uniform::SetCount(int cnt)
{
    m_count = cnt;
    return shared_from_this();
}

std::shared_ptr<Material::Uniform> Material::Uniform::SetTranspose(bool transpose)
{ 
    m_transpose = transpose;
    return shared_from_this();
}

void* Material::Uniform::GetData() 
{ 
    if(!m_data)
    {
        CORE_ASSERT(m_type != Type::Unknown, "You should tell me the datatype first!");
        _AllocateData();
    }
    
    return m_data.get();
}

void Material::Uniform::UpdateData(const void* data)
{
    CORE_ASSERT(data, "Material::Uniform::UpdateData: nullptr!");
    int size = m_count*_TypeSize();
    memcpy(GetData(), data, size);
}

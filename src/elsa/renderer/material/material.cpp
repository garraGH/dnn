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
#include "glm/gtx/string_cast.hpp"

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

std::string Material::Uniform::TypeString() const
{
#define ENUM2STRING(x) case Type::x: return #x;
    switch(m_type)
    {
        ENUM2STRING(Float1)
        ENUM2STRING(Float2)
        ENUM2STRING(Float3)
        ENUM2STRING(Float4)

        ENUM2STRING(Int1)
        ENUM2STRING(Int2)
        ENUM2STRING(Int3)
        ENUM2STRING(Int4)

        ENUM2STRING(UInt1)
        ENUM2STRING(UInt2)
        ENUM2STRING(UInt3)
        ENUM2STRING(UInt4)

        ENUM2STRING(Mat2x2)
        ENUM2STRING(Mat2x3)
        ENUM2STRING(Mat2x4)
        ENUM2STRING(Mat3x2)
        ENUM2STRING(Mat3x3)
        ENUM2STRING(Mat3x4)
        ENUM2STRING(Mat4x2)
        ENUM2STRING(Mat4x3)
        ENUM2STRING(Mat4x4)
        default: return "Unknown";
    }
#undef ENUM2STRING
}

std::string Material::Uniform::DataString() const
{
    switch(m_type)
    {
        case Type::Float1: return glm::to_string(*reinterpret_cast<glm::vec1*>(m_data.get()));
        case Type::Float2: return glm::to_string(*reinterpret_cast<glm::vec2*>(m_data.get()));
        case Type::Float3: return glm::to_string(*reinterpret_cast<glm::vec3*>(m_data.get()));
        case Type::Float4: return glm::to_string(*reinterpret_cast<glm::vec4*>(m_data.get()));

        case Type::Int1:   return glm::to_string(*reinterpret_cast<glm::ivec1*>(m_data.get()));
        case Type::Int2:   return glm::to_string(*reinterpret_cast<glm::ivec2*>(m_data.get()));
        case Type::Int3:   return glm::to_string(*reinterpret_cast<glm::ivec3*>(m_data.get()));
        case Type::Int4:   return glm::to_string(*reinterpret_cast<glm::ivec4*>(m_data.get()));

        case Type::UInt1:  return glm::to_string(*reinterpret_cast<glm::uvec1*>(m_data.get()));
        case Type::UInt2:  return glm::to_string(*reinterpret_cast<glm::uvec2*>(m_data.get()));
        case Type::UInt3:  return glm::to_string(*reinterpret_cast<glm::uvec3*>(m_data.get()));
        case Type::UInt4:  return glm::to_string(*reinterpret_cast<glm::uvec4*>(m_data.get()));

        case Type::Mat2x2: return glm::to_string(*reinterpret_cast<glm::mat2x2*>(m_data.get()));
        case Type::Mat2x3: return glm::to_string(*reinterpret_cast<glm::mat2x3*>(m_data.get()));
        case Type::Mat2x4: return glm::to_string(*reinterpret_cast<glm::mat2x4*>(m_data.get()));
        case Type::Mat3x2: return glm::to_string(*reinterpret_cast<glm::mat3x2*>(m_data.get()));
        case Type::Mat3x3: return glm::to_string(*reinterpret_cast<glm::mat3x3*>(m_data.get()));
        case Type::Mat3x4: return glm::to_string(*reinterpret_cast<glm::mat3x4*>(m_data.get()));
        case Type::Mat4x2: return glm::to_string(*reinterpret_cast<glm::mat4x2*>(m_data.get()));
        case Type::Mat4x3: return glm::to_string(*reinterpret_cast<glm::mat4x3*>(m_data.get()));
        case Type::Mat4x4: return glm::to_string(*reinterpret_cast<glm::mat4x4*>(m_data.get()));
        default: return "Unknown";
    }
}

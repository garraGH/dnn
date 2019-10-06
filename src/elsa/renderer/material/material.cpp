/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : material.cpp
* author      : Garra
* time        : 2019-10-05 21:55:48
* description : 
*
============================================*/


#include "material.h"

int Material::Attribute::_TypeSize() const
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
        default: CORE_ASSERT(false, "Unknown AttributeType!"); return 0;
    }
}

void Material::Attribute::_Save(const void* data)
{
    int size = m_count*_TypeSize();
    m_data = new unsigned char[size];
    memcpy((void*)m_data, data, size);
}                                  

void Material::Attribute::_Clear()
{
    switch(m_type)
    {
        case Type::Float1:
        case Type::Float2:
        case Type::Float3:
        case Type::Float4:
        case Type::Mat2x2:
        case Type::Mat2x3:
        case Type::Mat2x4:
        case Type::Mat3x2:
        case Type::Mat3x3:
        case Type::Mat3x4:
        case Type::Mat4x2:
        case Type::Mat4x3:
        case Type::Mat4x4: delete[] (float*)m_data; break;
        case Type::Int1: 
        case Type::Int2: 
        case Type::Int3: 
        case Type::Int4:   delete[] (int*)m_data; break;
        case Type::UInt1:
        case Type::UInt2:
        case Type::UInt3:
        case Type::UInt4:  delete[] (unsigned int*)m_data; break;
        default: break;
    }
}

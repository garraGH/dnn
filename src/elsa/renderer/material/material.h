/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : material.h
* author      : Garra
* time        : 2019-10-05 21:55:48
* description : 
*
============================================*/


#pragma once
#include <string>
#include <vector>
#include <memory>
#include <map>
#include "../shader/shader.h"
#include "logger.h"
#include "../../core.h"

class Material
{
public:
    class Attribute
    {
    public:
        enum class Type { Unknown = -1, Float1, Float2, Float3, Float4, Int1, Int2, Int3, Int4, UInt1, UInt2, UInt3, UInt4,   Mat2x2, Mat2x3, Mat2x4, Mat3x2, Mat3x3, Mat3x4, Mat4x2, Mat4x3, Mat4x4 };

        Attribute(const std::string& name, const void* data, Type type, int count=1, bool transpose=false) : m_name(name), m_type(type), m_count(count), m_transpose(transpose) { _Save(data); }

        Type GetType() const { return m_type; }
        const void* GetData() const { return m_data.get(); }
        const std::string& GetName() { return m_name; }
        int GetCount() const { return m_count; }
        bool NeedTranspose() const { return m_transpose; }

    protected:
        int _TypeSize() const;
        void _Save(const void* data);

    private:
        const std::string m_name;
        std::shared_ptr<void> m_data;
        Type m_type;
        int m_count = 1;
        bool m_transpose = false;
    };

public:
    Material(const std::string& name="undefined") : m_name(name) {}
    const std::string& GetName() const { return m_name; }
    void SetAttribute(const std::string& name, const std::shared_ptr<Attribute>& attribute) { m_attributes[name] = attribute; m_dirty = true; }
    virtual void Bind(const std::shared_ptr<Shader>& shader) = 0;
    static std::shared_ptr<Material> Create(const std::string& name);

protected:
    std::shared_ptr<Shader> m_shader = nullptr;
    std::map< const std::string, std::shared_ptr<Attribute> > m_attributes;
    bool m_dirty = true;

private:
    const std::string m_name;
};

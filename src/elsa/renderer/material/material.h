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

class Material : public Asset, public std::enable_shared_from_this<Material>
{
public:
    class Attribute : public Asset, public std::enable_shared_from_this<Attribute>
    {
    public:
        enum class Type { Unknown = -1, Float1, Float2, Float3, Float4, Int1, Int2, Int3, Int4, UInt1, UInt2, UInt3, UInt4,   Mat2x2, Mat2x3, Mat2x4, Mat3x2, Mat3x3, Mat3x4, Mat4x2, Mat4x3, Mat4x4 };

        Attribute(const std::string& name) : Asset(name) {}
        static std::shared_ptr<Attribute> Create(const std::string& name) { return std::make_shared<Attribute>(name); }

        std::shared_ptr<Attribute> Set(Type type, const void* data, int cnt=1, bool transpose=false) { m_type = type; m_count = cnt; m_transpose = transpose; _Save(data); return shared_from_this(); }
        std::shared_ptr<Attribute> SetType(Type type) { m_type = type; return shared_from_this(); }
        std::shared_ptr<Attribute> SetData(const void* data) { _Save(data); return shared_from_this(); }
        std::shared_ptr<Attribute> SetCount(int cnt) { m_count = cnt; return shared_from_this(); }
        std::shared_ptr<Attribute> SetTranspose(bool transpose) { m_transpose = transpose; return shared_from_this(); }
        void UpdateData(const void* data);

        Type GetType() const { return m_type; }
        void* GetData() const { return m_data.get(); }
        int GetCount() const { return m_count; }
        bool NeedTranspose() const { return m_transpose; }

    protected:
        int _TypeSize() const;
        void _Save(const void* data);

    private:
        std::shared_ptr<void> m_data; // auto delete (void*)data
        Type m_type;
        int m_count = 1;
        bool m_transpose = false;
    };

public:
    Material(const std::string& name="unnamed") : Asset(name) {}
    std::shared_ptr<Material> Set(const std::string& name, const std::shared_ptr<Attribute>& attribute) { m_attributes[name] = attribute; m_dirty = true; return shared_from_this(); }
    
    virtual void Bind(const std::shared_ptr<Shader>& shader) = 0;
    static std::shared_ptr<Material> Create(const std::string& name);

protected:
    std::shared_ptr<Shader> m_shader = nullptr;
    std::map< const std::string, std::shared_ptr<Attribute> > m_attributes;
    bool m_dirty = true;
};

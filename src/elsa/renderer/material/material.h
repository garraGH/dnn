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
#include "../texture/texture.h"
#include "logger.h"
#include "../../core.h"
#include "../buffer/buffer.h"

class Material : public Asset, public std::enable_shared_from_this<Material>
{
public:
    class Uniform : public Asset, public std::enable_shared_from_this<Uniform>
    {
    public:
        enum class Type { Unknown = -1, Float1, Float2, Float3, Float4, Int1, Int2, Int3, Int4, UInt1, UInt2, UInt3, UInt4, Mat2x2, Mat2x3, Mat2x4, Mat3x2, Mat3x3, Mat3x4, Mat4x2, Mat4x3, Mat4x4 };

        Uniform(const std::string& name) : Asset(name) {}
        static std::shared_ptr<Uniform> Create(const std::string& name) { return std::make_shared<Uniform>(name); }

        std::shared_ptr<Uniform> Set(Type type, int cnt=1, const void* data=nullptr, bool transpose=false);
        std::shared_ptr<Uniform> SetType(Type type);
        std::shared_ptr<Uniform> SetData(const void* data);
        std::shared_ptr<Uniform> SetCount(int cnt);
        std::shared_ptr<Uniform> SetTranspose(bool transpose);
        void UpdateData(const void* data);

        void* GetData();
        Type GetType() const { return m_type; }
        int GetCount() const { return m_count; }
        bool NeedTranspose() const { return m_transpose; }

        static std::string GetTypeName() { return "Material::Uniform"; }
    
        std::string TypeString() const;
        std::string DataString() const;

    protected:
        int _TypeSize() const;
        void _AllocateData();
        void _Save(const void* data);

    private:
        std::shared_ptr<void> m_data = nullptr; // auto delete (void*)data
        Type m_type = Type::Unknown;
        int m_count = 1;
        bool m_transpose = false;
    };

public:
    Material(const std::string& name="unnamed") : Asset(name) {}
    std::shared_ptr<Material> SetUniform(const std::string& name, const std::shared_ptr<Uniform>& uniform) { m_uniforms[name] = uniform; m_dirty = true; return shared_from_this(); }
    std::shared_ptr<Material> SetTexture(const std::string& name, const std::shared_ptr<Texture>& tex) { m_textures[name] = tex; m_dirty = true; return shared_from_this(); }
    std::shared_ptr<Material> SetUniformBuffer(const std::string& name, const std::shared_ptr<UniformBuffer>& ub) { m_uniformBuffers[name] = ub; m_dirty = true; return shared_from_this(); }
    
    virtual void Bind(const std::shared_ptr<Shader>& shader) = 0;
    static std::string GetTypeName() { return "Material"; }
    static std::shared_ptr<Material> Create(const std::string& name);

protected:
    std::shared_ptr<Shader> m_shader = nullptr;
    std::map< const std::string, std::shared_ptr<Uniform> > m_uniforms;
    std::map< const std::string, std::shared_ptr<Texture> > m_textures;
    std::map< const std::string, std::shared_ptr<UniformBuffer> > m_uniformBuffers;

    bool m_dirty = true;
};

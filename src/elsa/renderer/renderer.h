/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : renderer.h
* author      : Garra
* time        : 2019-10-01 23:14:38
* description : 
*
============================================*/


#pragma once
#include "camera/camera.h"
#include "transform/transform.h"
#include "buffer/buffer.h"
#include "mesh/mesh.h"
#include "material/material.h"
#include "shader/shader.h"
#include "texture/texture2d.h"
#include "viewport/viewport.h"

class Renderer
{
public:
    class API
    {
    public:
        enum Type
        {
            UNKOWN = 0, 
            OpenGL = 1, 
            Vulcan = 2,
            DirectX9 = 3, 
            DirectX11 = 4, 
            Directx12 = 5, 
            Metal = 6
        };

        virtual void SetBackgroundColor(float r, float g, float b, float a) = 0;
        virtual void SetViewport(float left, float bottom, float right, float top) = 0;
        virtual void DrawIndexed(const std::shared_ptr<BufferArray>& bufferArray) = 0;

        static inline Type GetType() { return s_type; }

    protected:
        static Type s_type;
    };

    class Command
    {
    public:
        static inline void SetBackgroundColor(float r, float g, float b, float a) { s_api->SetBackgroundColor(r, g, b, a); }
        static inline void SetViewport(float left, float bottom, float right, float top) { s_api->SetViewport(left, bottom, right, top); }
        static inline void DrawIndexed(const std::shared_ptr<BufferArray>& bufferArray) { s_api->DrawIndexed(bufferArray); }
    };

    class Element : public Asset, public std::enable_shared_from_this<Element>
    {
    public:
        Element(const std::string& name) : Asset(name) {}
        static std::shared_ptr<Element> Create(const std::string& name) { return std::make_shared<Element>(name); }
        std::shared_ptr<Element> Set(const std::shared_ptr<Mesh>& mesh, const std::shared_ptr<Material>& material)
        {
            m_mesh = mesh;
            m_material = material;
            return shared_from_this();
        }

        virtual std::string GetTypeName() const { return "Renderer::Element"; }
        std::shared_ptr<Element> SetMesh(const std::shared_ptr<Mesh>& mesh) { m_mesh = mesh; return shared_from_this(); }
        std::shared_ptr<Element> SetMaterial(const std::shared_ptr<Material>& material) { m_material = material; return shared_from_this(); }
        void RenderedBy(const std::shared_ptr<Shader>& shader);

    private:
        std::shared_ptr<Mesh> m_mesh = nullptr;
        std::shared_ptr<Material> m_material = nullptr;
    };

    template<typename T>
    class Assets
    {
    public:
        static Assets<T>& Instance() { return s_instance; }

        bool Exist(const std::string& name) { return m_assets.find(name) != m_assets.end(); }
        std::shared_ptr<T> Create(const std::string& name="unnamed") 
        {
            if(Exist(name))
            {
                CORE_WARN("Assets::Add: asset already exist! "+m_assets[name]->GetTypeName()+" - "+name);
                return m_assets[name];
            }
            std::shared_ptr<T> asset = T::Create(name);
            m_assets[name] = asset;
            return asset;
        }

        void Add(const std::shared_ptr<T>& asset) 
        {
            const std::string& name = asset->GetName();
            if(Exist(name))
            {
                CORE_WARN("Assets::Add: asset already exist! "+asset->GetTypeName()+" - "+name);
            }
            m_assets[name] = asset;
        }

        std::shared_ptr<T>& Get(const std::string& name) { CORE_ASSERT(Exist(name), "Assets::Get: asset not found! "+name); return m_assets[name]; }
    private:
        static Assets<T> s_instance;
        std::map<std::string, std::shared_ptr<T>> m_assets;
    };

    class Resources
    {
    public:
        template<typename T>
        static std::shared_ptr<T> Create(const std::string& name="unnamed") { return Assets<T>::Instance().Create(name); }
        template<typename T>
        static std::shared_ptr<T>& Get(const std::string& name) { return Assets<T>::Instance().Get(name); }
        template<typename T>
        static void Add(const std::shared_ptr<T>& asset) { Assets<T>::Instance().Add(asset); }
    };

public:

    inline static API::Type GetAPIType() { return s_api->GetType(); }
    static void SetAPIType(API::Type apiType);
    static void SetBackgroundColor(float r, float g, float b, float a) { Command::SetBackgroundColor(r, g, b, a); }

    static void BeginScene(const std::shared_ptr<Viewport>& viewport);
    static void Submit(const std::shared_ptr<Element>& rendererElement, const std::shared_ptr<Shader>& shader);
    static void Submit(const std::string& nameOfElement, const std::string& nameOfShader);
    static void EndScene() {}


private:
    static std::unique_ptr<API> s_api;
    static std::shared_ptr<Camera> s_camera;
};

template<typename T>
Renderer::Assets<T> Renderer::Assets<T>::s_instance;

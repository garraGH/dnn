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
#include "buffer/buffer.h"
#include "shader/shader.h"
#include "transform/transform.h"
#include "camera/camera.h"
#include "material/material.h"
#include "mesh/mesh.h"


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
        virtual void DrawIndexed(const std::shared_ptr<BufferArray>& bufferArray) = 0;

        static inline Type GetType() { return s_type; }

    protected:
        static Type s_type;
    };

    class Command
    {
    public:
        static inline void SetBackgroundColor(float r, float g, float b, float a) { s_api->SetBackgroundColor(r, g, b, a); }
        static inline void DrawIndexed(const std::shared_ptr<BufferArray>& bufferArray) { s_api->DrawIndexed(bufferArray); }
    };

    class Element
    {
    public:
        Element(const std::shared_ptr<Mesh>& mesh, const std::shared_ptr<Material>& material) : m_mesh(mesh), m_material(material) {}
        void SetMesh(const std::shared_ptr<Mesh>& mesh) { m_mesh = mesh; }
        void SetMaterial(const std::shared_ptr<Material>& material) { m_material = material; }
        void SetTransform(const std::shared_ptr<Transform>& transform) { m_mesh->SetTransform(transform); }
        void SetMaterialAttribute(const std::string& name, const std::shared_ptr<Material::Attribute>& attribute) { m_material->SetAttribute(name, attribute); }
        void Draw(const std::shared_ptr<Shader>& shader);

    private:
        std::shared_ptr<Mesh> m_mesh = nullptr;
        std::shared_ptr<Material> m_material = nullptr;
    };

public:

    inline static API::Type GetAPIType() { return s_api->GetType(); }
    static void SetAPIType(API::Type apiType);

    static void BeginScene(std::shared_ptr<Camera>& camera) { s_camera = camera; }
    static void EndScene() {}
    static void SetBackgroundColor(float r, float g, float b, float a) { Command::SetBackgroundColor(r, g, b, a); }
    static void Submit(const std::shared_ptr<Element>& rendererElement, const std::shared_ptr<Shader>& shader);


private:
    static std::unique_ptr<API> s_api;
    static std::shared_ptr<Camera> s_camera;
};

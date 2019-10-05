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
        Element(const std::shared_ptr<BufferArray>& bufferArray=nullptr, const std::shared_ptr<Shader>& shader=nullptr, const std::shared_ptr<Transform>& transform=nullptr);
        void SetBufferArray(const std::shared_ptr<BufferArray>& bufferArray) { m_bufferArray = bufferArray; }
        void SetShader(const std::shared_ptr<Shader>& shader) { m_shader = shader; }
        void SetTransform(const std::shared_ptr<Transform>& transform) { m_transform = transform; }

        void Draw();

    private:
        std::shared_ptr<BufferArray> m_bufferArray = nullptr;
        std::shared_ptr<Shader> m_shader = nullptr;
        std::shared_ptr<Transform> m_transform = nullptr;
    };

public:

    inline static API::Type GetAPIType() { return s_api->GetType(); }
    static void SetAPIType(API::Type apiType);

    static void BeginScene(std::shared_ptr<Camera>& camera) { s_camera = camera; }
    static void EndScene() {}
    static void SetBackgroundColor(float r, float g, float b, float a) { Command::SetBackgroundColor(r, g, b, a); }
    static void Submit(const std::shared_ptr<Element>& rendererElement);


private:
    static std::unique_ptr<API> s_api;
    static std::shared_ptr<Camera> s_camera;
};

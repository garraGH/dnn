/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : application.cpp
* author      : Garra
* time        : 2019-09-24 10:44:51
* description : 
*
============================================*/


#include <stdio.h>
#include "glad/gl.h"
#include "application.h"
#include "logger.h"
#include "timer_cpu.h"
#include "core.h"
#include "../input/input.h"
#include "../renderer/renderer.h"
#include "../renderer/shader/shader_glsl.h"
#include "../renderer/buffer/buffer_opengl.h"
#include "../renderer/camera/camera_orthographic.h"
#include "glm/glm.hpp"
#include "glm/gtx/string_cast.hpp"

#define BIND_EVENT_CALLBACK(x) std::bind(&Application::x,  this, std::placeholders::_1)
Application* Application::s_instance = nullptr;

Application::Application()
    : m_running(true)
{
    CORE_ASSERT(!s_instance, "Application already exist!");
    s_instance = this;
    Renderer::SetAPIType(Renderer::API::OpenGL);

    m_window = std::unique_ptr<Window>(Window::Create(WindowsProps("Elsa", 1000, 1000)));
    m_window->SetEventCallback(BIND_EVENT_CALLBACK(OnEvent));

    m_layerImGui = new ImGuiLayer;
    PushOverlay(m_layerImGui);
    
    m_camera = std::make_shared<OrthographicCamera>(OrthographicCamera(-2, +2, -2, +2));



    float vertices[3*7] = 
    {
        -0.5f, -0.5f, 0.0f, 255, 0, 0, 255,  
        +0.5f, -0.5f, 0.0f, 0, 255, 0, 255, 
        +0.0f, +0.8f, 0.0f, 0, 0, 255, 255
    };

//     glEnable(GL_BLEND);
//     glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    Buffer::Layout layoutVextex = 
    {
        { Buffer::Element::DataType::Float3, "a_Position", false }, 
        { Buffer::Element::DataType::Int4, "a_Color", true }
    };
    m_vertexBuffer.reset(Buffer::CreateVertex(sizeof(vertices), vertices));
    m_vertexBuffer->SetLayout(layoutVextex);

    unsigned char indices[3] = { 0, 1, 2 };
    m_indexBuffer.reset(Buffer::CreateIndex(sizeof(indices), indices));
    Buffer::Layout layoutIndex;
    Buffer::Element e(Buffer::Element::DataType::UChar);
    layoutIndex.Push(e);
    m_indexBuffer->SetLayout(layoutIndex);

    std::string srcVertex = R"(
        #version 460 core
        layout(location = 0) in vec3 a_Position;
        layout(location = 1) in vec4 a_Color;
        uniform mat4 u_ViewProjection;
        out vec4 v_Position;
        out vec4 v_Color;

        void main()
        {
            gl_Position = u_ViewProjection*vec4(a_Position+0.2, 1.0f);
            v_Position = gl_Position;
            v_Color = a_Color;
        }
    )";

    std::string srcFragment = R"(
        #version 460 core
        in vec4 v_Position;
        out vec4 color;
        in vec4 v_Color;
        void main()
        {
//             color = v_Position;
            color = v_Color;
        }
    )";

    m_shader.reset(Shader::Create(srcVertex, srcFragment));

//     CORE_INFO("a_Position location: {}", glad_glGetAttribLocation(m_shader->ID(), "a_Position"));
//     CORE_INFO("a_Color location: {}", glad_glGetAttribLocation(m_shader->ID(), "a_Color"));
//     CORE_INFO("u_ViewProjection location: {}", glad_glGetUniformLocation(m_shader->ID(), "u_ViewProjection"));
//     const glm::mat4& vp = m_camera->GetViewProjectionMatrix();
//     CORE_INFO("VP: {}", glm::to_string(vp));

    m_bufferArrayTri.reset(BufferArray::Create());
    m_bufferArrayTri->SetShader(m_shader);
    m_bufferArrayTri->AddVertexBuffer(m_vertexBuffer);
    m_bufferArrayTri->SetIndexBuffer(m_indexBuffer);


    unsigned short indices_quad[] = { 0, 1, 2, 0, 2, 3 };
    std::shared_ptr<Buffer> indexBuffer_quad(Buffer::CreateIndex(sizeof(indices_quad), indices_quad));
    Buffer::Layout layoutIndex_quad = 
    {
        { Buffer::Element::DataType::UShort }
    };
    indexBuffer_quad->SetLayout(layoutIndex_quad);


    float vertices_quad[4*3] = 
    {
        -0.5f, -0.5f, 0.0f,
        +0.5f, -0.5f, 0.0f,
        +0.5f, +0.5f, 0.0f,
        -0.5f, +0.5f, 0.0f
    };

    float colors_quad[4*4] = 
    {
        0, 0, 0, 255, 
        255, 0, 0, 255, 
        255, 255, 255, 255, 
        0, 255, 0, 255 
    };

    Buffer::Layout layoutVextex_quad = 
    {
        { Buffer::Element::DataType::Float3, "a_Position", false }, 
    };
    Buffer::Layout layoutColor_quad = 
    {
        { Buffer::Element::DataType::Int4, "a_Color", true }
    };

    std::shared_ptr<Buffer> vertexBuffer_quad(Buffer::CreateVertex(sizeof(vertices_quad), vertices_quad));
    std::shared_ptr<Buffer> colorBuffer_quad(Buffer::CreateVertex(sizeof(colors_quad), colors_quad));

    vertexBuffer_quad->SetLayout(layoutVextex_quad);
    colorBuffer_quad->SetLayout(layoutColor_quad);


    m_bufferArrayQuad.reset(BufferArray::Create());
    m_bufferArrayQuad->SetShader(m_shader);
    m_bufferArrayQuad->AddVertexBuffer(vertexBuffer_quad);
    m_bufferArrayQuad->AddVertexBuffer(colorBuffer_quad);
    m_bufferArrayQuad->SetIndexBuffer(indexBuffer_quad);
    

#define REGISTER_KEY_PRESSED_FUNCTION(key) m_keyPressed[#key[0]] = std::bind(&Application::_OnKeyPressed_##key, this, std::placeholders::_1)
#define REGISTER_KEY_RELEASED_FUNCTION(key) m_keyReleased[#key[0]] = std::bind(&Application::_OnKeyReleased_##key, this)
    REGISTER_KEY_PRESSED_FUNCTION(a);
    REGISTER_KEY_PRESSED_FUNCTION(R);
    REGISTER_KEY_RELEASED_FUNCTION(q);
    REGISTER_KEY_RELEASED_FUNCTION(Q);
#undef REGISTER_KEY_PRESSED_FUNCTION
#undef REGISTER_KEY_RELEASED_FUNCTION
}

Application::~Application()
{
    CORE_TRACE("Application destructed.");
}

void Application::OnEvent(Event& e)
{
    CORE_TRACE("{0}", e);
    EventDispatcher ed(e);
#define DISPATCH(event) ed.Dispatch<event>(BIND_EVENT_CALLBACK(_On##event))
    DISPATCH(WindowCloseEvent);
    DISPATCH(KeyPressedEvent);
    DISPATCH(KeyReleasedEvent);
#undef DISPATCH
    for(auto it = m_layerStack.end(); it != m_layerStack.begin();)
    {
        (*--it)->OnEvent(e);
        if(e.IsHandled())
        {
            break;
        }
    }
}

#define _ON(event) bool Application::_On##event(event& e)
_ON(WindowCloseEvent)
{
    INFO("CLOSED");
    m_running = false;
    return true;
}

_ON(KeyPressedEvent)
{
    std::function<bool(int)> fn = m_keyPressed[e.GetKeyCode()];
    return fn == nullptr? false : fn(e.GetRepeatCount());
}

_ON(KeyReleasedEvent)
{
    std::function<bool()> fn = m_keyReleased[e.GetKeyCode()];
    return fn == nullptr? false : fn();
}

#undef _ON

void Application::Run()
{
    TimerCPU t("Application::Run");
    INFO("Application::Run\n");
    WindowResizeEvent e(1280, 720);
    if(e.IsCategory(EC_Application))
    {
        TRACE(e);
    }
    if(e.IsCategory(EC_Input))
    {
        TRACE(e);
    }

    while(m_running)
    {
        m_camera->SetPosition(glm::vec3(0.5, 0.5, 0.0));
        m_camera->SetRotation(glm::vec3(0, 0, 90));
        Renderer::BeginScene(m_camera);
        Renderer::SetBackgroundColor(0.1, 0.1, 0.1, 1);
        Renderer::Submit(m_bufferArrayQuad);
        Renderer::Submit(m_bufferArrayTri);
        Renderer::EndScene();

        for(Layer* layer : m_layerStack)
        {
            layer->OnUpdate();
        }

        m_layerImGui->Begin();
        for(Layer* layer : m_layerStack)
        {
            layer->OnImGuiRender();
        }
        m_layerImGui->End();

        m_window->OnUpdate();

    }
}

bool Application::_OnKeyPressed_a(int repeatCount)
{
    INFO("a pressed: {}", repeatCount);
    return true;
}

bool Application::_OnKeyPressed_R(int repeatCount)
{
    INFO("R pressed: {}", repeatCount);
    return true;
}

bool Application::_OnKeyReleased_q()
{
    INFO("q released");
    m_running = false;
    return true;
}

bool Application::_OnKeyReleased_Q()
{
    INFO("Q released");
    m_running = false;
    return true;
}

void Application::PushLayer(Layer* layer)
{
    layer->OnAttach();
    m_layerStack.PushLayer(layer);
}

void Application::PushOverlay(Layer* layer)
{
    layer->OnAttach();
    m_layerStack.PushOverlay(layer);
}

#undef BIND_EVENT_CALLBACK


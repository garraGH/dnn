/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : src/elsa/layer/layer_example.c
* author      : Garra
* time        : 2019-10-04 00:03:27
* description : 
*
============================================*/


#include "layer_example.h"
#include "../renderer/shader/shader_glsl.h"
#include "../renderer/buffer/buffer_opengl.h"
#include "../renderer/camera/camera_orthographic.h"
#include "../input/input.h"
#include "glm/gtx/string_cast.hpp"


ExampleLayer::ExampleLayer()
{
    Renderer::SetAPIType(Renderer::API::OpenGL);
    m_camera = std::make_shared<OrthographicCamera>(OrthographicCamera(-2, +2, -2, +2));
    m_transform = std::make_shared<Transform>(Transform(glm::vec3(0.0f), glm::vec3(0.0f), glm::vec3(0.1f)));

    float vertices[3*7] = 
    {
        -0.5f, -0.5f, 0.0f, 255, 0, 0, 255,  
        +0.5f, -0.5f, 0.0f, 0, 255, 0, 255, 
        +0.0f, +0.8f, 0.0f, 0, 0, 255, 255
    };

    Buffer::Layout layoutVextex = 
    {
        { Buffer::Element::DataType::Float3, "a_Position", false }, 
        { Buffer::Element::DataType::Int4, "a_Color", true }
    };
    std::shared_ptr<Buffer> vertexBuffer(Buffer::CreateVertex(sizeof(vertices), vertices));
    vertexBuffer->SetLayout(layoutVextex);

    unsigned char indices[3] = { 0, 1, 2 };
    std::shared_ptr<Buffer> indexBuffer(Buffer::CreateIndex(sizeof(indices), indices));
    Buffer::Layout layoutIndex;
    Buffer::Element e(Buffer::Element::DataType::UChar);
    layoutIndex.Push(e);
    indexBuffer->SetLayout(layoutIndex);

    std::string srcVertex = R"(
        #version 460 core
        layout(location = 0) in vec3 a_Position;
        layout(location = 1) in vec4 a_Color;
        uniform mat4 u_ViewProjection;
        uniform mat4 u_Transform;
        out vec4 v_Position;
        out vec4 v_Color;

        void main()
        {
            gl_Position = u_ViewProjection*u_Transform*vec4(a_Position, 1.0f);
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

    std::shared_ptr<Shader> shader(Shader::Create(srcVertex, srcFragment));
    std::shared_ptr<BufferArray> bufferArrayTri = std::shared_ptr<BufferArray>(BufferArray::Create());
    bufferArrayTri->AddVertexBuffer(vertexBuffer);
    bufferArrayTri->SetIndexBuffer(indexBuffer);


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


    std::shared_ptr<BufferArray> bufferArrayQuad = std::shared_ptr<BufferArray>(BufferArray::Create());
    bufferArrayQuad->AddVertexBuffer(vertexBuffer_quad);
    bufferArrayQuad->AddVertexBuffer(colorBuffer_quad);
    bufferArrayQuad->SetIndexBuffer(indexBuffer_quad);

    m_reTri = std::make_shared<Renderer::Element>(Renderer::Element(bufferArrayTri, shader));
    m_reQuad = std::make_shared<Renderer::Element>(Renderer::Element(bufferArrayQuad, shader));
}
    
void ExampleLayer::OnEvent(Event& e)
{
    TRACE("ExampleLayer: event {}", e);
}

void ExampleLayer::_UpdateCamera(float deltaTime)
{
    float distance = m_speedTranslate*deltaTime;
    float angle = m_speedRotate*deltaTime;


    if(Input::IsKeyPressed(KEY_LEFT) || Input::IsKeyPressed(KEY_S))
    {
        CORE_INFO("KEY_LEFT");
        m_camera->Translate({+distance, 0, 0});
    }
    if(Input::IsKeyPressed(KEY_RIGHT) || Input::IsKeyPressed(KEY_F))
    {
        m_camera->Translate({-distance, 0, 0});
    }
    if(Input::IsKeyPressed(KEY_UP) || Input::IsKeyPressed(KEY_E))
    {
        m_camera->Translate({0, -distance, 0});
    }
    if(Input::IsKeyPressed(KEY_DOWN) || Input::IsKeyPressed(KEY_D))
    {
        m_camera->Translate({0, +distance, 0});
    }

    if(Input::IsKeyPressed(KEY_J))
    {
        m_camera->Rotate({0, 0, +angle});
    }
    if(Input::IsKeyPressed(KEY_K))
    {
        m_camera->Rotate({0, 0, -angle});
    }
    if(Input::IsKeyPressed(KEY_P))
    {
        m_camera->Revert();
    }
}

void ExampleLayer::_UpdateQuads(float deltaTime)
{
    float displacement = m_speedTranslate*deltaTime;
    float degree = m_speedRotate*deltaTime;
    if(Input::IsKeyPressed(KEY_Z))
    {
        m_transform->Translate({-displacement, 0, 0});
    }
    if(Input::IsKeyPressed(KEY_V))
    {
        m_transform->Translate({+displacement, 0, 0});
    }
    if(Input::IsKeyPressed(KEY_X))
    {
        m_transform->Translate({0, -displacement, 0});
    }
    if(Input::IsKeyPressed(KEY_C))
    {
        m_transform->Translate({0, +displacement, 0});
    }
    if(Input::IsKeyPressed(KEY_W))
    {
        m_transform->Scale(-glm::vec3(0.001f));
    }
    if(Input::IsKeyPressed(KEY_R))
    {
        m_transform->Scale(+glm::vec3(0.001f));
    }
    if(Input::IsKeyPressed(KEY_H))
    {
        m_transform->Rotate({0, 0, +degree});
    }
    if(Input::IsKeyPressed(KEY_L))
    {
        m_transform->Rotate({0, 0, -degree});
    }

    for(int i=0; i<5; i++)
    {
        for(int j=0; j<5; j++)
        {
            m_reQuad->SetTransform(m_transform);
            Renderer::Submit(m_reQuad);
            m_transform->Translate({ 0.15f, 0, 0 });
        }
        m_transform->Translate({-0.15f*5, 0, 0});
        m_transform->Translate({0, +0.15f, 0});
    }
    m_transform->Translate({0, -0.15f*5, 0});
}

void ExampleLayer::_UpdateScene(float deltaTime)
{
    Renderer::BeginScene(m_camera);
    Renderer::SetBackgroundColor(0.1, 0.1, 0.1, 1);
    Renderer::Submit(m_reTri);
    _UpdateQuads(deltaTime);
    Renderer::EndScene();
}

void ExampleLayer::OnUpdate(float deltaTime)
{
    _UpdateCamera(deltaTime);
    _UpdateScene(deltaTime);
}

void ExampleLayer::OnImGuiRender()
{
    ImGui::Begin("ExampleLayer");
    ImGui::Button("ExampleLayer");
    ImGui::End();
}

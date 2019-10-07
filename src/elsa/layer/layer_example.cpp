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
#include "../renderer/mesh/mesh.h"
#include "../renderer/material/material.h"


ExampleLayer::ExampleLayer()
{
    Renderer::SetAPIType(Renderer::API::OpenGL);
    m_camera = std::make_shared<OrthographicCamera>(-2, +2, -2, +2);

    _PrepareAssets();

    m_reTri = std::make_shared<Renderer::Element>(Renderer::Assets<Mesh>::Instance().Get("mesh_tri"), Renderer::Assets<Material>::Instance().Get("mtr_tri"));
    m_reQuad = std::make_shared<Renderer::Element>(Renderer::Assets<Mesh>::Instance().Get("mesh_quad"), Renderer::Assets<Material>::Instance().Get("mtr_quad"));
}

void ExampleLayer::_PrepareAssets()
{
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
    std::shared_ptr<Buffer> vertexBuffer_tri = Buffer::CreateVertex(sizeof(vertices), vertices);
    vertexBuffer_tri->SetLayout(layoutVextex);

    unsigned char indices[3] = { 0, 1, 2 };
    std::shared_ptr<Buffer> indexBuffer_tri = Buffer::CreateIndex(sizeof(indices), indices);
    Buffer::Layout layoutIndex;
    Buffer::Element e(Buffer::Element::DataType::UChar);
    layoutIndex.Push(e);
    indexBuffer_tri->SetLayout(layoutIndex);


    unsigned short indices_quad[] = { 0, 1, 2, 0, 2, 3 };
    std::shared_ptr<Buffer> indexBuffer_quad = Buffer::CreateIndex(sizeof(indices_quad), indices_quad);
    Buffer::Layout layoutIndex_quad = 
    {
        { Buffer::Element::DataType::UShort }
    };
    indexBuffer_quad->SetLayout(layoutIndex_quad);


    float position_quad[4*3] = 
    {
        -0.5f, -0.5f, 0.0f,
        +0.5f, -0.5f, 0.0f,
        +0.5f, +0.5f, 0.0f,
        -0.5f, +0.5f, 0.0f
    };

    float color_quad[4*4] = 
    {
        0, 0, 0, 255, 
        255, 0, 0, 255, 
        255, 255, 255, 255, 
        0, 255, 0, 255 
    };

    Buffer::Layout layoutPosition_quad = 
    {
        { Buffer::Element::DataType::Float3, "a_Position", false }, 
    };
    Buffer::Layout layoutColor_quad = 
    {
        { Buffer::Element::DataType::Int4, "a_Color", true }
    };


    std::shared_ptr<Buffer> positionBuffer_quad = Buffer::CreateVertex(sizeof(position_quad), position_quad);
    std::shared_ptr<Buffer> colorBuffer_quad = Buffer::CreateVertex(sizeof(color_quad), color_quad);
    positionBuffer_quad->SetLayout(layoutPosition_quad);
    colorBuffer_quad->SetLayout(layoutColor_quad);

    using MA = Material::Attribute;
    Renderer::Assets<MA>::Instance().Add(std::make_shared<MA>("red", glm::value_ptr(glm::vec4(1, 0.1, 0.2, 1)), MA::Type::Float4));
    Renderer::Assets<MA>::Instance().Add(std::make_shared<MA>("green", glm::value_ptr(glm::vec4(0.1, 1, 0.2, 1)), MA::Type::Float4));
    Renderer::Assets<MA>::Instance().Add(std::make_shared<MA>("blue", glm::value_ptr(glm::vec4(0.1, 0.2, 1, 1)), MA::Type::Float4));

    std::shared_ptr<Mesh> meshTri = Renderer::Assets<Mesh>::Instance().Create("mesh_tri");
    std::shared_ptr<Mesh> meshQuad = Renderer::Assets<Mesh>::Instance().Create("mesh_quad");
    std::shared_ptr<Material> materialTri = Renderer::Assets<Material>::Instance().Create("mtr_tri");
    std::shared_ptr<Material> materialQuad = Renderer::Assets<Material>::Instance().Create("mtr_quad");

    meshTri->SetIndexBuffer(indexBuffer_tri);
    meshTri->AddVertexBuffer(vertexBuffer_tri);
    meshQuad->SetIndexBuffer(indexBuffer_quad);
    meshQuad->AddVertexBuffer(positionBuffer_quad);
    meshQuad->AddVertexBuffer(colorBuffer_quad);

    materialTri->SetAttribute("u_Color", Renderer::Assets<MA>::Instance().Get("green"));
    materialQuad->SetAttribute("u_Color", Renderer::Assets<MA>::Instance().Get("red"));

    Renderer::Assets<Shader>::Instance().Create("basic")->LoadFile("/home/garra/study/dnn/assets/shader/basic.glsl");
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
        m_transformQuad->SetTranslation(glm::vec3(0));
        m_transformQuad->SetRotation(glm::vec3(0));
        m_transformQuad->SetScale(glm::vec3(0.1f));
    }
}

void ExampleLayer::_UpdateQuads(float deltaTime)
{
    float displacement = m_speedTranslate*deltaTime;
    float degree = m_speedRotate*deltaTime;
    if(Input::IsKeyPressed(KEY_Z))
    {
        m_transformQuad->Translate({-displacement, 0, 0});
    }
    if(Input::IsKeyPressed(KEY_V))
    {
        m_transformQuad->Translate({+displacement, 0, 0});
    }
    if(Input::IsKeyPressed(KEY_X))
    {
        m_transformQuad->Translate({0, -displacement, 0});
    }
    if(Input::IsKeyPressed(KEY_C))
    {
        m_transformQuad->Translate({0, +displacement, 0});
    }
    if(Input::IsKeyPressed(KEY_W))
    {
        m_transformQuad->Scale(-glm::vec3(0.001f));
    }
    if(Input::IsKeyPressed(KEY_R))
    {
        m_transformQuad->Scale(+glm::vec3(0.001f));
    }
    if(Input::IsKeyPressed(KEY_H))
    {
        m_transformQuad->Rotate({0, 0, +degree});
    }
    if(Input::IsKeyPressed(KEY_L))
    {
        m_transformQuad->Rotate({0, 0, -degree});
    }

    using MA = Material::Attribute;
    std::shared_ptr<MA> red = Renderer::Assets<MA>::Instance().Get("red");
    std::shared_ptr<MA> blue = Renderer::Assets<MA>::Instance().Get("blue");

    for(int i=0; i<5; i++)
    {
        for(int j=0; j<5; j++)
        {
            m_reQuad->SetTransform(m_transformQuad);
            m_reQuad->SetMaterialAttribute("u_Color", (i+j)%2? red : blue);
            Renderer::Submit(m_reQuad, Renderer::Assets<Shader>::Instance().Get("basic"));
            m_transformQuad->Translate({ 0.15f, 0, 0 });
        }
        m_transformQuad->Translate({-0.15f*5, 0, 0});
        m_transformQuad->Translate({0, +0.15f, 0});
    }
    m_transformQuad->Translate({0, -0.15f*5, 0});
}

void ExampleLayer::_UpdateTri(float deltaTime)
{
    m_reTri->SetTransform(m_transformTri);
    m_reTri->SetMaterialAttribute("u_Color", Renderer::Assets<Material::Attribute>::Instance().Get("green"));
    Renderer::Submit(m_reTri, Renderer::Assets<Shader>::Instance().Get("basic"));
}

void ExampleLayer::_UpdateScene(float deltaTime)
{
    Renderer::BeginScene(m_camera);
    Renderer::SetBackgroundColor(0.1, 0.1, 0.1, 1);
    _UpdateTri(deltaTime);
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
    using MA = Material::Attribute;
    std::shared_ptr<MA> red = Renderer::Assets<MA>::Instance().Get("red");
    std::shared_ptr<MA> green = Renderer::Assets<MA>::Instance().Get("green");

    ImGui::Begin("ExampleLayer");
    ImGui::Button("ExampleLayer");
    ImGui::ColorPicker4("color_tri", (float*)green->GetData());
    ImGui::ColorPicker4("color_quad", (float*)red->GetData());
    ImGui::End();
}

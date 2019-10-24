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
#include "glm/gtc/type_ptr.hpp"
#include "glm/gtx/string_cast.hpp"
#include "imgui.h"

std::shared_ptr<ExampleLayer> ExampleLayer::Create()
{
    return std::make_shared<ExampleLayer>();
}

ExampleLayer::ExampleLayer()
    : Layer("ExampleLayer")
{
    _PrepareResources();
}

void ExampleLayer::_PrepareResources()
{
    float vertices[3*7] = 
    {
        -0.5f, -0.5f, -1.0f, 255, 0, 0, 255,  
        +0.5f, -0.5f, -1.0f, 0, 255, 0, 255, 
        +0.0f, +0.5f, -1.0f, 0, 0, 255, 255
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
        -0.5f, -0.5f, -1.0f,
        +0.5f, -0.5f, -1.0f,
        +0.5f, +0.5f, -1.0f,
        -0.5f, +0.5f, -1.0f
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

    Renderer::Resources::Create<Mesh>("mesh_tri")->Set(indexBuffer_tri, {vertexBuffer_tri});
    Renderer::Resources::Create<Mesh>("mesh_quad")->Set(indexBuffer_quad, {positionBuffer_quad, colorBuffer_quad});

    using MA = Material::Attribute;
    Renderer::Resources::Create<MA>("red")->Set(MA::Type::Float4, 1, glm::value_ptr(glm::vec4(1, 0.1, 0.2, 1)));
    Renderer::Resources::Create<MA>("green")->Set(MA::Type::Float4, 1, glm::value_ptr(glm::vec4(0.1, 1, 0.2, 1)));
    Renderer::Resources::Create<MA>("blue")->Set(MA::Type::Float4, 1, glm::value_ptr(glm::vec4(0.1, 0.2, 1, 1)));
    Renderer::Resources::Create<Material>("mtr_tri")->Set("u_Color", Renderer::Resources::Get<MA>("green"));
    Renderer::Resources::Create<Material>("mtr_quad")->Set("u_Color", Renderer::Resources::Get<MA>("red"));
    Renderer::Resources::Create<Shader>("Basic")->LoadFromFile("/home/garra/study/dnn/assets/shader/Basic.glsl");
    Renderer::Resources::Create<Renderer::Element>("ele_tri")->Set(Renderer::Resources::Get<Mesh>("mesh_tri"), Renderer::Resources::Get<Material>("mtr_tri"));
    Renderer::Resources::Create<Renderer::Element>("ele_quad")->Set(Renderer::Resources::Get<Mesh>("mesh_quad"), Renderer::Resources::Get<Material>("mtr_quad"));

    Renderer::Resources::Create<Transform>("tf_tri");
    Renderer::Resources::Create<Transform>("tf_quad")->SetScale(glm::vec3(0.1f));
}
    
void ExampleLayer::OnEvent(Event& e)
{
    m_viewport->OnEvent(e);
}

void ExampleLayer::_UpdateViewport(float deltaTime)
{
    m_viewport->OnUpdate(deltaTime);
}

void ExampleLayer::_TransformQuads(float deltaTime)
{
    float displacement = m_speedTranslate*deltaTime;
    float degree = m_speedRotate*deltaTime;
    std::shared_ptr<Transform> tf_quad = Renderer::Resources::Get<Transform>("tf_quad");

    if(Input::IsKeyPressed(KEY_j))
    {
        tf_quad->Translate({-displacement, 0, 0});
    }
    if(Input::IsKeyPressed(KEY_J))
    {
        tf_quad->Translate({+displacement, 0, 0});
    }
    if(Input::IsKeyPressed(KEY_k))
    {
        tf_quad->Translate({0, -displacement, 0});
    }
    if(Input::IsKeyPressed(KEY_K))
    {
        tf_quad->Translate({0, +displacement, 0});
    }
    if(Input::IsKeyPressed(KEY_l))
    {
        tf_quad->Translate({0, 0, -displacement});
    }
    if(Input::IsKeyPressed(KEY_L))
    {
        tf_quad->Translate({0, 0, +displacement});
    }

    if(Input::IsKeyPressed(KEY_u))
    {
        tf_quad->Scale(-glm::vec3(0.001f));
    }
    if(Input::IsKeyPressed(KEY_U))
    {
        tf_quad->Scale(+glm::vec3(0.001f));
    }
    if(Input::IsKeyPressed(KEY_i))
    {
        tf_quad->Rotate({0, 0, -degree});
    }
    if(Input::IsKeyPressed(KEY_I))
    {
        tf_quad->Rotate({0, 0, +degree});
    }

    if(Input::IsKeyPressed(KEY_B))
    {
        tf_quad->Revert();
    }
}

void ExampleLayer::_UpdateQuads(float deltaTime)
{
    _TransformQuads(deltaTime);

    using MA = Material::Attribute;
    std::shared_ptr<MA> red = Renderer::Resources::Get<MA>("red");
    std::shared_ptr<MA> blue = Renderer::Resources::Get<MA>("blue");
    std::shared_ptr<Material> mtr_quad = Renderer::Resources::Get<Material>("mtr_quad");
    std::shared_ptr<Mesh> mesh_tri = Renderer::Resources::Get<Mesh>("mesh_tri");
    std::shared_ptr<Mesh> mesh_quad = Renderer::Resources::Get<Mesh>("mesh_quad");

    std::shared_ptr<Transform> tf_quad = Renderer::Resources::Get<Transform>("tf_quad");
    for(int i=0; i<5; i++)
    {
        for(int j=0; j<5; j++)
        {
            mesh_quad->SetTransform(tf_quad);
            mtr_quad->Set("u_Color", (i+j)%2? red : blue);
            Renderer::Submit("ele_quad", "Basic");
            tf_quad->Translate({ 0.15f, 0, 0 });
        }
        tf_quad->Translate({-0.15f*5, 0, 0});
        tf_quad->Translate({0, +0.15f, 0});
    }
    tf_quad->Translate({0, -0.15f*5, 0});
}

void ExampleLayer::_UpdateTri(float deltaTime)
{
    Renderer::Resources::Get<Mesh>("mesh_tri")->SetTransform(Renderer::Resources::Get<Transform>("tf_tri"));
    Renderer::Resources::Get<Material>("mtr_tri")->Set("u_Color", Renderer::Resources::Get<Material::Attribute>("green"));
    Renderer::Submit("ele_tri", "Basic");
}

void ExampleLayer::_UpdateScene(float deltaTime)
{
    Renderer::BeginScene(m_viewport);
    Renderer::SetBackgroundColor(0.1, 0.1, 0.1, 1);
    _UpdateTri(deltaTime);
    _UpdateQuads(deltaTime);
    Renderer::EndScene();
}

void ExampleLayer::OnUpdate(float deltaTime)
{
    _UpdateViewport(deltaTime);
    _UpdateScene(deltaTime);
}

void ExampleLayer::OnImGuiRender()
{
    using MA = Material::Attribute;
    std::shared_ptr<MA> red = Renderer::Resources::Get<MA>("red");
    std::shared_ptr<MA> green = Renderer::Resources::Get<MA>("green");

    m_viewport->OnImGuiRender();

    ImGui::Begin("ExampleLayer");
    ImGui::Button("ExampleLayer");
    ImGui::ColorPicker4("color_tri", (float*)green->GetData());
    ImGui::ColorPicker4("color_quad", (float*)red->GetData());
    ImGui::End();
}

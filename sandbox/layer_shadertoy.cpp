/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : layer_shadertoy.cpp
* author      : Garra
* time        : 2019-10-10 14:26:49
* description : 
*
============================================*/


#include "layer_shadertoy.h"
#include "glm/gtc/type_ptr.hpp"

std::shared_ptr<ShaderToyLayer> ShaderToyLayer::Create()
{
    return std::make_shared<ShaderToyLayer>();
}

ShaderToyLayer::ShaderToyLayer()
    : Layer("ShaderToyLayer")
{
    _PrepareResources();
}

void ShaderToyLayer::_PrepareResources()
{
    unsigned char indices[] = { 0, 1, 2, 0, 2, 3 };
    Buffer::Layout layoutIndex = { { Buffer::Element::DataType::UChar } };
    std::shared_ptr<Buffer> indexBuffer = Buffer::CreateIndex(sizeof(indices), indices);
    indexBuffer->SetLayout(layoutIndex);


    float vertices[4*3] = 
    {
        -1.0f, -1.0f, 0.0f,
        +1.0f, -1.0f, 0.0f,
        +1.0f, +1.0f, 0.0f,
        -1.0f, +1.0f, 0.0f
    };
    Buffer::Layout layoutVertex = { { Buffer::Element::DataType::Float3, "a_Position", false }, };
    std::shared_ptr<Buffer> vertexBuffer = Buffer::CreateVertex(sizeof(vertices), vertices);
    vertexBuffer->SetLayout(layoutVertex);

    using MA = Material::Attribute;
    Renderer::Resources::Create<Mesh>("shadercanvas")->Set(indexBuffer, {vertexBuffer});
    Renderer::Resources::Create<Shader>("flatcolor")->LoadFromFile("/home/garra/study/dnn/assets/shader/flatcolor.glsl");
    Renderer::Resources::Create<Shader>("shapes")->LoadFromFile("/home/garra/study/dnn/assets/shader/shapes.glsl");
    Renderer::Resources::Create<Shader>("shapingfunctions")->LoadFromFile("/home/garra/study/dnn/assets/shader/shapingfunctions.glsl");
    std::shared_ptr<MA> ma_flatcolor = Renderer::Resources::Create<MA>("flatcolor")->Set(MA::Type::Float4, glm::value_ptr(glm::vec4(0.8, 0.1, 0.2, 1.0)));
    std::shared_ptr<MA> ma_resolution = Renderer::Resources::Create<MA>("resolution")->Set(MA::Type::Float2, glm::value_ptr(glm::vec2(1000, 1000)));

    Renderer::Resources::Create<Material>("resolution")->Set("u_Resolution", ma_resolution);
    Renderer::Resources::Create<Material>("flatcolor")->Set("u_Color", ma_flatcolor);
    Renderer::Resources::Create<Material>("shadercanvas")->Set("u_Color", ma_flatcolor)->Set("u_Resolution", ma_resolution);
    Renderer::Resources::Create<Renderer::Element>("shadercanvas")->Set(Renderer::Resources::Get<Mesh>("shadercanvas"), Renderer::Resources::Get<Material>("shadercanvas"));
}

void ShaderToyLayer::OnUpdate(float deltaTime)
{
    Renderer::BeginScene(m_cameraController->GetCamera());
//     Renderer::Submit("shadercanvas", "flatcolor");
    Renderer::Submit("shadercanvas", "shapes");
//     Renderer::Submit("shadercanvas", "shapingfunctions");
    Renderer::EndScene();
}

void ShaderToyLayer::OnEvent(Event& e)
{

}

void ShaderToyLayer::OnImGuiRender()
{

}

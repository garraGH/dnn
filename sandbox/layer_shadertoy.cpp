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
        -0.5f, -0.5f, 0.0f,
        +0.5f, -0.5f, 0.0f,
        +0.5f, +0.5f, 0.0f,
        -0.5f, +0.5f, 0.0f
    };
    Buffer::Layout layoutVertex = { { Buffer::Element::DataType::Float3, "a_Position", false }, };
    std::shared_ptr<Buffer> vertexBuffer = Buffer::CreateVertex(sizeof(vertices), vertices);
    vertexBuffer->SetLayout(layoutVertex);

    Renderer::Resources::Create<Mesh>("shader_canvas")->Set(indexBuffer, {vertexBuffer});
    Renderer::Resources::Create<Shader>("flat_color")->LoadFromFile("/home/garra/study/dnn/assets/shader/flat_color.glsl");
    Renderer::Resources::Create<Material::Attribute>("flat_color")->Set(Material::Attribute::Type::Float4, glm::value_ptr(glm::vec4(0.8, 0.1, 0.2, 1.0)));
    Renderer::Resources::Create<Material>("flat_color")->Set("u_Color", Renderer::Resources::Get<Material::Attribute>("flat_color"));
    Renderer::Resources::Create<Renderer::Element>("shader_canvas")->Set(Renderer::Resources::Get<Mesh>("shader_canvas"), Renderer::Resources::Get<Material>("flat_color"));
}

void ShaderToyLayer::OnUpdate(float deltaTime)
{
    Renderer::BeginScene(m_cameraController->GetCamera());
    Renderer::Submit("shader_canvas", "flat_color");
    Renderer::EndScene();
}

void ShaderToyLayer::OnEvent(Event& e)
{

}

void ShaderToyLayer::OnImGuiRender()
{

}

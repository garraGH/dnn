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
    for(int i=ShaderToy::Type::Unknown+1; i<ShaderToy::Type::Last; i++)
    {
        _Register(ShaderToy::Type(i));
    }
    _PrepareResources();
}

void ShaderToyLayer::_Register(ShaderToy::Type type)
{
    m_shaderToys[type] = ShaderToy::Create(type);
}

void ShaderToyLayer::_PrepareResources()
{
    unsigned char indices[] = { 0, 1, 2, 0, 2, 3 };
    Buffer::Layout layoutIndex = { { Buffer::Element::DataType::UChar } };
    std::shared_ptr<Buffer> indexBuffer = Buffer::CreateIndex(sizeof(indices), indices);
    indexBuffer->SetLayout(layoutIndex);


    float vertices[4*5] = 
    {
        -1.0f, -1.0f, 0.0f, 0.0f, 0.0f, 
        +1.0f, -1.0f, 0.0f, 1.0f, 0.0f, 
        +1.0f, +1.0f, 0.0f, 1.0f, 1.0f, 
        -1.0f, +1.0f, 0.0f, 0.0f, 1.0f
    };
    Buffer::Layout layoutVertex = { { Buffer::Element::DataType::Float3, "a_Position", false }, { Buffer::Element::DataType::Float2, "a_TexCoord" }};
    std::shared_ptr<Buffer> vertexBuffer = Buffer::CreateVertex(sizeof(vertices), vertices);
    vertexBuffer->SetLayout(layoutVertex);

    std::shared_ptr<Elsa::Mesh> msCanvas = Renderer::Resources::Create<Elsa::Mesh>("Canvas")->Set(indexBuffer, {vertexBuffer});


    m_canvas = Renderer::Resources::Create<Renderer::Element>("Canvas")->Set(msCanvas, m_shaderToys[m_toyType]->GetMaterial(), m_shaderToys[m_toyType]->GetShader());
}

void ShaderToyLayer::OnUpdate(float deltaTime)
{
    Renderer::BeginScene(m_viewport);
    Renderer::Submit(m_canvas);
    m_shaderToys[m_toyType]->OnUpdate(deltaTime);
    m_viewport->OnUpdate(deltaTime);
    Renderer::EndScene();
}

void ShaderToyLayer::OnEvent(Event& e)
{
    m_shaderToys[m_toyType]->OnEvent(e);
    m_viewport->OnEvent(e);
}

void ShaderToyLayer::OnImGuiRender()
{
    ImGui::Begin("ShaderToyLayer");
    {
        for(auto& toy : m_shaderToys)
        {
            ImGui::SameLine();
            _CreateRadioButtonOf(toy.second);
        }

        ImGui::Separator();
        m_shaderToys[m_toyType]->OnImGuiRender();
    }
    ImGui::End();

    m_viewport->OnImGuiRender();
}

void ShaderToyLayer::_CreateRadioButtonOf(const std::shared_ptr<ShaderToy>& toy)
{
    if(ImGui::RadioButton(toy->GetName().c_str(), m_toyType == toy->GetType())) 
    {                                                                 
        m_toyType = toy->GetType();                            
        m_canvas->SetMaterial(m_shaderToys[m_toyType]->GetMaterial());
    }                                                                 
}


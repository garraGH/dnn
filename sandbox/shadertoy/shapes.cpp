/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : shapes.cpp
* author      : Garra
* time        : 2019-10-13 09:50:21
* description : 
*
============================================*/


#include "shapes.h"
#include "glm/gtc/type_ptr.hpp"

std::shared_ptr<Shapes> Shapes::Create()
{
    return std::make_shared<Shapes>();
}

Shapes::Shapes()
{
    m_type = Type::Shapes;
    _PrepareResources();
}

void Shapes::_PrepareResources()
{
    using MA = Material::Attribute;
    std::shared_ptr<MA> maResolution = Renderer::Resources::Create<MA>("Resolution")->Set(MA::Type::Float2, glm::value_ptr(glm::vec2(1000, 1000)));
    Renderer::Resources::Create<Shader>("Shapes")->LoadFromFile("/home/garra/study/dnn/assets/shader/Shapes.glsl");
    Renderer::Resources::Create<Material>("Shapes")->Set("u_Resolution", maResolution);
}

std::shared_ptr<Material> Shapes::GetMaterial() const
{
    return Renderer::Resources::Get<Material>("Shapes");
}

std::shared_ptr<Shader> Shapes::GetShader() const
{
    return Renderer::Resources::Get<Shader>("Shapes");
}

void Shapes::OnImGuiRender()
{

}

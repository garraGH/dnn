/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : renderer.cpp
* author      : Garra
* time        : 2019-10-01 23:14:45
* description : 
*
============================================*/


#include <array>
#include "renderer.h"
#include "api/api_opengl.h"

std::shared_ptr<Camera> Renderer::s_camera = nullptr;
std::unique_ptr<Renderer::API> Renderer::s_api = std::make_unique<OpenGLAPI>();

Renderer::API::Type Renderer::API::s_type = API::Type::UNKOWN;


void Renderer::SetAPIType(API::Type apiType)
{
    if(s_api && apiType==s_api->GetType())
    {
        return;
    }

    switch(apiType)
    {
        case API::OpenGL: s_api = std::make_unique<OpenGLAPI>(); break;
        default: CORE_ASSERT(false, "Renderer::SetAPIType: API is currently not supported!");
    }
}

void Renderer::BeginScene(const std::shared_ptr<Viewport>& viewport, const std::shared_ptr<FrameBuffer>& frameBuffer)
{
    Command::SetFrameBuffer(frameBuffer);
    Command::SetViewport(viewport);
    s_camera = viewport->GetCamera();
}

void Renderer::Submit(const std::string& nameOfElement, const std::string& nameOfShader, unsigned int nInstances)
{
    Resources::Get<Element>(nameOfElement)->RenderedBy(Resources::Get<Shader>(nameOfShader), nInstances);
}

void Renderer::Submit(const std::shared_ptr<Renderer::Element>& rendererElement, const std::shared_ptr<Shader>& shader, unsigned int nInstances) 
{
    rendererElement->RenderedBy(shader, nInstances);
}

void Renderer::Element::RenderedBy(const std::shared_ptr<Shader>& shader, unsigned int nInstances)
{
    CORE_ASSERT(shader, "Renderer::Element::RenderedBy: Must Bind a shader to render.");

    shader->Bind();
//     if(s_camera->IsDirty())
//     {
//         shader->SetWorld2ClipMatrix(s_camera->World2Clip());
//     }
// 
    if(m_material)
    {
        m_material->Bind(shader);
    }

    m_mesh->Bind(shader);
    Renderer::Command::DrawElements(m_mesh->GetBufferArray(), nInstances);
}



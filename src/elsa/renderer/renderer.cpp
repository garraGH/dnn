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
#include "osdialog.h"

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
    CORE_ASSERT(rendererElement, "Renderer::Submit: null rendererElement!");
    rendererElement->RenderedBy(shader, nInstances);
}

void Renderer::Submit(const std::shared_ptr<Renderer::Element>& element)
{
    CORE_ASSERT(element, "Renderer::Submit: null element!");
    element->Render();
}

void Renderer::Submit(const std::string& nameOfElement)
{
    Renderer::Resources::Get<Renderer::Element>(nameOfElement)->Render();
}

void Renderer::Element::RenderedBy(const std::shared_ptr<Shader>& shader, unsigned int nInstances)
{
    CORE_ASSERT(m_mesh, "Renderer::Element{}::RenderedBy: Must provide a mesh to render.", m_name);
    CORE_ASSERT(shader, "Renderer::Element{}::RenderedBy: Must Bind a shader to render.", m_name);

//     INFO("Renderer::Element::RenderedBy: 1");
    shader->Bind();
//     INFO("Renderer::Element::RenderedBy: 2");
    if(m_material)
    {
        m_material->Bind(shader);
    }

//     INFO("Renderer::Element::RenderedBy: 3");
    m_mesh->Bind(shader);
//     INFO("Renderer::Element::RenderedBy: 4");
    Renderer::Command::DrawElements(m_mesh->GetBufferArray(), nInstances);
}

void Renderer::Element::Render()
{
    CORE_ASSERT(m_mesh, "Renderer::Element{}::Render: Must provide a mesh to render.", m_name);
    CORE_ASSERT(m_shader, "Renderer::Element{}::Render: Must Bind a shader to render.", m_name);

    if(!m_bVisible)
    {
        return;
    }

//     INFO("Element({})::Render: nInstance({})", m_name, m_nInstance);
    m_shader->Bind();

    if(m_material)
    {
        m_material->Bind(m_shader);
    }
    m_mesh->Bind(m_shader);
    Renderer::Command::DrawElements(m_mesh->GetBufferArray(), m_nInstance);
}


void Renderer::Element::_UpdateTexture(std::shared_ptr<Texture>& texture)
{
    char* filename = osdialog_file(OSDIALOG_OPEN, "/home/garra/study/dnn/assets/texture", nullptr, nullptr);
    if(filename)
    {
        texture->Reload(filename);
        delete[] filename;
    }
}

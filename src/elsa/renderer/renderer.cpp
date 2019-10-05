/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : renderer.cpp
* author      : Garra
* time        : 2019-10-01 23:14:45
* description : 
*
============================================*/


#include "renderer.h"
#include "api/api_opengl.h"

std::shared_ptr<Camera> Renderer::s_camera = nullptr;
std::unique_ptr<Renderer::API> Renderer::s_api = nullptr;
Renderer::API::Type Renderer::API::s_type = API::Type::UNKOWN;

void Renderer::SetAPIType(API::Type apiType)
{
    if(s_api && apiType==s_api->GetType())
    {
        return ;
    }

    switch(apiType)
    {
        case API::OpenGL: s_api.reset(new OpenGLAPI()); break;
        default: CORE_ASSERT(false, "Renderer::SetAPIType: API is currently not supported!");
    }
}


void Renderer::Submit(const std::shared_ptr<Renderer::Element>& rendererElement) 
{
    rendererElement->Draw();
}

Renderer::Element::Element(const std::shared_ptr<BufferArray>& bufferArray, const std::shared_ptr<Shader>& shader, const std::shared_ptr<Transform>& transform)
    : m_bufferArray(bufferArray)
    , m_shader(shader)
    , m_transform(transform)
{
    if(!m_transform)
    {
        m_transform = std::make_shared<Transform>(Transform());
    }
}

void Renderer::Element::Draw()
{
    CORE_ASSERT(m_bufferArray&&m_shader, "Renderer::Element must have at least one BufferArray&&Shader to be drawn!");
    m_shader->Bind();
    m_shader->UploadUniformMat4("u_ViewProjection", s_camera->GetViewProjectionMatrix());
    m_shader->UploadUniformMat4("u_Transform", m_transform->GetTransformMatrx());

    m_bufferArray->UsedbyShader(m_shader);
    m_bufferArray->Bind();
    Renderer::Command::DrawIndexed(m_bufferArray);
}

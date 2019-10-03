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

void Renderer::Submit(const std::shared_ptr<BufferArray>& bufferArray) 
{
    auto shader = bufferArray->GetShader();
    shader->Bind();
    shader->UploadUniformMat4("u_ViewProjection", s_camera->GetViewProjectionMatrix());
    Command::DrawIndexed(bufferArray);
}

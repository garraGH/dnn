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

std::unique_ptr<Renderer::API> Renderer::s_api = std::make_unique<OpenGLAPI>(OpenGLAPI());
Renderer::API::Type Renderer::API::s_type = API::Type::OpenGL;

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

/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : dnn.cpp
* author      : Garra
* time        : 2019-09-24 10:48:21
* description : 
*
============================================*/


#define LOG_TRACE
#include <stdio.h>
#include "layer_example.h"
#include "layer_shadertoy.h"
#include "logger.h"



class DNN : public Application
{
public:
    DNN()
    {
        Renderer::SetAPIType(Renderer::API::OpenGL);

        PushLayer(ExampleLayer::Create());
//         PushLayer(ShaderToyLayer::Create());
    }

    ~DNN()
    {

    }

protected:
private:
};

std::unique_ptr<Application> CreateApplication()
{
    return std::make_unique<DNN>();
}

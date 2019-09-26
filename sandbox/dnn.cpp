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
#include "elsa.h"
#include "logger.h"

#include <stdio.h>

class ExampleLayer : public Layer
{
public:
    void OnEvent(Event& e) override
    {
        TRACE("ExampleLayer: event {}", e);
    }
    void OnUpdate() override
    {
        TRACE("ExampleLayer OnUpdate.");
    }
};

class DNN : public Application
{
public:
    DNN()
    {
        PushLayer(new ExampleLayer());
    }

    ~DNN()
    {

    }

protected:
private:
};

Application* CreateApplication()
{
    return new DNN();
}

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
#include "logger.h"



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

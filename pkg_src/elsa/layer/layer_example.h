/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : layer_example.h
* author      : Garra
* time        : 2019-09-26 22:15:00
* description : 
*
============================================*/


#pragma once
#include "layer.h"
#include "logger.h"

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

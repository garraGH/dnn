/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : input_glfw.h
* author      : Garra
* time        : 2019-09-28 17:47:21
* description : 
*
============================================*/


#pragma once
#include "input.h"

class GLFWInput : public Input
{
protected:
    virtual bool _IsKeyPressed(KeyCode keyCode) override;
    virtual bool _IsMouseButtonPressed(MouseButtonCode mouseButtonCode) override;
    virtual std::pair<float, float> _GetMousePosition() override;
    virtual float _GetMouseX() override;
    virtual float _GetMouseY() override;
};

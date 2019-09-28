/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : input.h
* author      : Garra
* time        : 2019-09-28 17:47:20
* description : 
*
============================================*/


#pragma once

#include <utility>
#include "codes_key.h"
#include "codes_mouse.h"

class Input
{
public:
    inline static bool IsKeyPressed(KeyCode keyCode) { return s_input->_IsKeyPressed(keyCode); } 
    inline static bool IsMouseButtonPressed(MouseButtonCode mouseButtonCode) { return s_input->_IsMouseButtonPressed(mouseButtonCode); }
    inline static std::pair<float, float> GetMousePosition() { return s_input->_GetMousePosition(); }
    inline static float GetMouseX() { return s_input->_GetMouseX(); }
    inline static float GetMouseY() { return s_input->_GetMouseY(); }


protected:
    virtual bool _IsKeyPressed(KeyCode keyCode) = 0;
    virtual bool _IsMouseButtonPressed(MouseButtonCode mouseButtonCode) = 0;
    virtual std::pair<float, float> _GetMousePosition() = 0;
    virtual float _GetMouseX() = 0;
    virtual float _GetMouseY() = 0;

private:
    static Input* s_input;
};

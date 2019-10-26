/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : /home/garra/study/dnn/pkg_src/elsa/input/input_glfw.c
* author      : Garra
* time        : 2019-09-28 17:53:58
* description : 
*
============================================*/


#include "input_glfw.h"
#include "GLFW/glfw3.h"
#include "../app/application.h"

Input* Input::s_input = new GLFWInput;

bool GLFWInput::_IsKeyPressed(KeyCode keyCode)
{
    auto window = static_cast<GLFWwindow*>(Application::GetInstance()->GetWindow()->GetNativeWindow());
    if(KEY_a <= keyCode && keyCode <= KEY_z)    // a-z
    {
        keyCode = KeyCode(keyCode-0x20);        // to A-Z
        auto state = glfwGetKey(window, keyCode);
        bool pressed =  state == GLFW_PRESS || state == GLFW_REPEAT;
        if(!pressed)                            // key is not pressed
        {
            return false;
        }
        bool shiftPressed = _IsKeyPressed(KEY_LEFT_SHIFT) || _IsKeyPressed(KEY_RIGHT_SHIFT);
        return _IsKeyPressed(KEY_TAB)? shiftPressed : !shiftPressed; // return true if TAB&SHIFT both pressed or both released
    }

    auto state = glfwGetKey(window, keyCode);
    bool pressed =  state == GLFW_PRESS || state == GLFW_REPEAT;
    if(!pressed) // key is not pressed
    {
        return false;
    }

    if(keyCode<KEY_A || keyCode>KEY_Z) // not A-Z return true directly
    {
        return true;
    }

    bool shiftPressed = _IsKeyPressed(KEY_LEFT_SHIFT) || _IsKeyPressed(KEY_RIGHT_SHIFT);
    return _IsKeyPressed(KEY_TAB)? !shiftPressed : shiftPressed; // return true if TAB or SHIFT is pressed
}

bool GLFWInput::_IsMouseButtonPressed(MouseButtonCode mouseButtonCode)
{
    auto window = static_cast<GLFWwindow*>(Application::GetInstance()->GetWindow()->GetNativeWindow());
    auto state = glfwGetMouseButton(window, mouseButtonCode);
    return state == GLFW_PRESS;
}

std::pair<float, float> GLFWInput::_GetMousePosition()
{
    auto window = static_cast<GLFWwindow*>(Application::GetInstance()->GetWindow()->GetNativeWindow());
    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);
    return { (float)xpos, (float)ypos };
}


float GLFWInput::_GetMouseX()
{
    auto[x, y] = _GetMousePosition();
    return x;
}

float GLFWInput::_GetMouseY()
{
    auto[x, y] = _GetMousePosition();
    return y;
}


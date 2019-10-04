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
    auto window = static_cast<GLFWwindow*>(Application::Get()->GetWindow()->GetNativeWindow());
    auto state = glfwGetKey(window, keyCode);
    return state == GLFW_PRESS || state == GLFW_REPEAT;
}

bool GLFWInput::_IsMouseButtonPressed(MouseButtonCode mouseButtonCode)
{
    auto window = static_cast<GLFWwindow*>(Application::Get()->GetWindow()->GetNativeWindow());
    auto state = glfwGetMouseButton(window, mouseButtonCode);
    return state == GLFW_PRESS;
}

std::pair<float, float> GLFWInput::_GetMousePosition()
{
    auto window = static_cast<GLFWwindow*>(Application::Get()->GetWindow()->GetNativeWindow());
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


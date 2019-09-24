/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : entrypoint.cpp
* author      : Garra
* time        : 2019-09-24 11:20:47
* description : 
*
============================================*/

#include "application.h"

int main(int argc, char** argv)
{
    Application* app = CreateApplication();
    app->Run();
    delete app;

    return 0;
}


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
#include "logger.h"

int main(int argc, char** argv)
{
    Logger::Init();

    Application* app = CreateApplication();
    app->Run();
    delete app;

    Logger::Close();
    return 0;
}


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
#include "logger.h"
#include "timer.h"
#include "elsa.h"

#include <stdio.h>

class DNN : public Application
{
public:
    DNN()
    {

    }

    ~DNN()
    {

    }

    void Run() override
    {
        TimerCPU t("DNN:Run");
        CORE_TRACE("DNN:Run");
        CORE_WARN("DNN:Run");
        CORE_INFO("DNN:Run");
        CORE_ERROR("DNN:Run");
        CORE_CRITICAL("DNN:Run");
        TRACE("DNN:Run");
        WARN("DNN:Run");
        INFO("DNN:Run");
        ERROR("DNN:Run");
        CRITICAL("DNN:Run");
    }
protected:
private:
};

Application* CreateApplication()
{
    return new DNN();
}

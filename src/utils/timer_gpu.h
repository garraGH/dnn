/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : /home/garra/study/opengl/timer_gpu.h
* author      : Garra
* time        : 2019-09-03 20:36:57
* description : 
*
============================================*/

#include "timer.h"
#include <cuda_runtime.h>
#include <string>

#pragma once

class TimerGPU : public Timer
{
public:
    TimerGPU(const std::string& taskName="UnnamedTask");
    ~TimerGPU();

    virtual float GetElapsedTime() override;
    virtual float GetDeltaTime() override;

private:
    std::string m_taskName;
    cudaEvent_t m_beg = 0;
    cudaEvent_t m_pre = 0;
    cudaEvent_t m_now = 0;
};

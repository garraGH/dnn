/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : /home/garra/study/opengl/timer_gpu.h
* author      : Garra
* time        : 2019-09-03 20:36:57
* description : 
*
============================================*/

#include <cuda_runtime.h>

#pragma once

class TimerGPU : public Timer
{
public:
    TimerGPU(const char* msg);
    ~TimerGPU();

    virtual float timeUsed() override;
private:
    cudaEvent_t m_tbeg, m_tend;
};

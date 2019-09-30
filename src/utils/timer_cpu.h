/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : /home/garra/study/opengl/timer_cpu.h
* author      : Garra
* time        : 2019-09-03 20:36:57
* description : 
*
============================================*/

#include <chrono>
#include "timer.h"

#pragma once
class TimerCPU : public Timer
{
public:
    TimerCPU(const char* msg);
    ~TimerCPU();

    virtual float timeUsed() override;

private:
    std::chrono::time_point<std::chrono::high_resolution_clock, std::chrono::nanoseconds> m_beg;
};

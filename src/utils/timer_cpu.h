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
#include <string>
#include "timer.h"

#pragma once
class TimerCPU : public Timer
{
public:
    TimerCPU(const std::string& taskName="UnnamedTask");
    ~TimerCPU();

    virtual float GetElapsedTime() override;
    virtual float GetDeltaTime() override;

private:
    using TimePoint = std::chrono::time_point<std::chrono::high_resolution_clock, std::chrono::nanoseconds>;
    TimePoint m_beg;
    TimePoint m_pre;
    std::string m_taskName;
};

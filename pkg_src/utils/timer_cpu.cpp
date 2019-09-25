/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : timer_cpu.cpp
* author      : Garra
* time        : 2019-09-03 20:38:11
* description : 
*
============================================*/


#include "timer_cpu.h"

TimerCPU::TimerCPU(const char* msg)
    : Timer(msg)
{
    m_beg = std::chrono::high_resolution_clock::now();
}

float TimerCPU::timeUsed()
{
    if(m_done)
    {
        return m_timeUsed;
    }
    m_done = true;
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed_seconds = end-m_beg;
    m_timeUsed = elapsed_seconds.count() / 1e6;
    return m_timeUsed;
}

TimerCPU::~TimerCPU()
{
    timeUsed();
}

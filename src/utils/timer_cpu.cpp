/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : timer_cpu.cpp
* author      : Garra
* time        : 2019-09-03 20:38:11
* description : 
*
============================================*/


#include "logger.h"
#include "timer_cpu.h"


TimerCPU::TimerCPU(const std::string& taskName)
    : m_taskName(taskName)
{
    m_beg = std::chrono::high_resolution_clock::now();
    m_pre = m_beg;
}

float TimerCPU::GetElapsedTime()
{
    auto now = std::chrono::high_resolution_clock::now();
    auto elapsed_nanoseconds = (now-m_beg).count();
    auto elapsed_seconds = elapsed_nanoseconds/1e9;
    return elapsed_seconds;
}

float TimerCPU::GetDeltaTime()
{
    auto now = std::chrono::high_resolution_clock::now();
    auto elapsed_nanoseconds = (now-m_pre).count();
    auto elapsed_seconds = elapsed_nanoseconds/1e9;
    m_pre = now;
    return elapsed_seconds;
}

TimerCPU::~TimerCPU()
{
    INFO("( {} )TimeElapsed: {}s", m_taskName, GetElapsedTime());
}

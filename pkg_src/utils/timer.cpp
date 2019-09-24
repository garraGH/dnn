/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : timer.cpp
* author      : Garra
* time        : 2019-09-03 20:38:11
* description : 
*
============================================*/


#include "timer.h"
#include "logger.h"


Timer::Timer(const char* msg)
    : m_msg(msg)
    , m_timeUsed(0.0f)
    , m_done(false)
{
}

Timer::~Timer()
{
    INFO("time used ( {} ): {}ms", m_msg, m_timeUsed);
}




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

TimerGPU::TimerGPU(const char* msg)
    : Timer(msg)
{
    cudaEventCreate(&m_tbeg);
    cudaEventCreate(&m_tend);
    cudaEventRecord(m_tbeg, 0);
}

float TimerGPU::timeUsed()
{
    if(m_done)
    {
        return m_timeUsed;
    }
    m_done = true;
    cudaEventRecord(m_tend, 0);
    cudaEventSynchronize(m_tbeg);
    cudaEventSynchronize(m_tend);
    cudaEventElapsedTime(&m_timeUsed, m_tbeg, m_tend);
    return m_timeUsed;
}

TimerGPU::~TimerGPU()
{
    timeUsed();
}

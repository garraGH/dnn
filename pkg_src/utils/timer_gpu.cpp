/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : timer_gpu.cpp
* author      : Garra
* time        : 2019-09-03 20:38:11
* description : 
*
============================================*/


#include "timer_gpu.h"

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

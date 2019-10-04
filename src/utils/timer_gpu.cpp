/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : timer_gpu.cpp
* author      : Garra
* time        : 2019-09-03 20:38:11
* description : 
*
============================================*/


#include <logger.h>
#include "timer_gpu.h"

TimerGPU::TimerGPU(const std::string& taskName)
    : m_taskName(taskName)
{
    cudaEventCreate(&m_beg);
    cudaEventCreate(&m_pre);
    cudaEventCreate(&m_now);
    cudaEventRecord(m_beg, 0);
    cudaEventRecord(m_pre, 0);
}

float TimerGPU::GetElapsedTime()
{
    cudaEventRecord(m_now, 0);
    cudaEventSynchronize(m_beg);
    cudaEventSynchronize(m_now);
    float timeElapsed = 0.0f;
    cudaEventElapsedTime(&timeElapsed, m_beg, m_now);
    return timeElapsed;
}

float TimerGPU::GetDeltaTime()
{
    cudaEventRecord(m_now, 0);
    cudaEventSynchronize(m_pre);
    cudaEventSynchronize(m_now);
    float timeDelta = 0.0f;
    cudaEventElapsedTime(&timeDelta, m_pre, m_now);
    cudaEventRecord(m_pre, 0);
    return timeDelta;
}

TimerGPU::~TimerGPU()
{
    INFO("( {} )TimeElapsed: {}ms", m_taskName, GetElapsedTime());
}

/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : /home/garra/study/opengl/timer.h
* author      : Garra
* time        : 2019-09-03 20:36:57
* description : 
*
============================================*/

#include <chrono>
#include <cuda_runtime.h>

#pragma once
class Timer
{
public:
    Timer(const char* msg);
    virtual ~Timer();
    virtual float timeUsed()  = 0;
protected:
    const char* m_msg;
    float m_timeUsed;
    bool m_done;
};

class TimerCPU : public Timer
{
public:
    TimerCPU(const char* msg);
    ~TimerCPU();

    virtual float timeUsed() override;

private:
    std::chrono::time_point<std::chrono::high_resolution_clock, std::chrono::nanoseconds> m_beg;
};


class TimerGPU : public Timer
{
public:
    TimerGPU(const char* msg);
    ~TimerGPU();

    virtual float timeUsed() override;
private:
    cudaEvent_t m_tbeg, m_tend;
};

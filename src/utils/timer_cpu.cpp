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

bool TimerCPU::s_enableProfile = false;
uint32_t TimerCPU::s_numFramesTobeProfiled = -1;
uint32_t TimerCPU::s_curFrameProfiled = 0;

void TimerCPU::EnableProfile(bool b)
{
    s_enableProfile = b;
}

void TimerCPU::SetFramesTobeProfiled(int n)
{
    s_numFramesTobeProfiled = n;
    s_curFrameProfiled = 1;
}

void TimerCPU::IncreaseFrame()
{
    s_curFrameProfiled++;
}

bool TimerCPU::NeedProfile()
{
    return s_enableProfile;
}

TimerCPU::TimerCPU(const std::string& taskName, bool enableProfile, Callback func)
    : m_taskName(taskName)
    , m_enableProfile(enableProfile)
    , m_func(func)
{
    m_beg = Clock::now();
    m_pre = m_beg;
}

TimerCPU::~TimerCPU()
{
    if(m_func)
    {
        m_func(m_taskName, GetElapsedTime());
    }
    INFO("( {} )TimeElapsed: {}s", m_taskName, GetElapsedTime());

    _Profile();
}

void TimerCPU::_Profile()
{
    if(!s_enableProfile)
    {
        return;
    }

    if(!m_enableProfile)
    {
        return;
    }

    if(s_curFrameProfiled>s_numFramesTobeProfiled)
    {
        return;
    }

    uint32_t tid = std::hash<std::thread::id>{}(std::this_thread::get_id());
    long long beg = std::chrono::time_point_cast<std::chrono::microseconds>(m_beg       ).time_since_epoch().count();
    long long end = std::chrono::time_point_cast<std::chrono::microseconds>(Clock::now()).time_since_epoch().count();
    Profile::Get().Write(m_taskName, tid, beg, end);
}

float TimerCPU::GetElapsedTime()
{
    auto now = Clock::now();
    auto elapsed_nanoseconds = (now-m_beg).count();
    auto elapsed_seconds = elapsed_nanoseconds/1e9;
    return elapsed_seconds;
}

float TimerCPU::GetDeltaTime()
{
    auto now = Clock::now();
    auto elapsed_nanoseconds = (now-m_pre).count();
    auto elapsed_seconds = elapsed_nanoseconds/1e9;
    m_pre = now;
    return elapsed_seconds;
}

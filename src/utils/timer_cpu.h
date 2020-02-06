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
#include "logger.h"
#include "profile.h"

#pragma once
class TimerCPU : public Timer
{
public:
    using Callback = std::function<void (const std::string&, float)>;
    TimerCPU(const std::string& taskName, bool enableProfile=false, Callback func=nullptr);
    ~TimerCPU();

    float GetElapsedTime() override;
    float GetDeltaTime() override;

    static void EnableProfile(bool b);
    static void SetFramesTobeProfiled(int n);
    static void IncreaseFrame();
    static bool NeedProfile();

protected:
    void _Profile();

private:
    std::string m_taskName;

    using Clock = std::chrono::high_resolution_clock;
    using NanoSec = std::chrono::nanoseconds;
    using TimePoint = std::chrono::time_point<Clock, NanoSec>;

    TimePoint m_beg;
    TimePoint m_pre;

    bool m_enableProfile = false;
    Callback m_func = nullptr;
    static bool s_enableProfile;
    static uint32_t s_numFramesTobeProfiled;
    static uint32_t s_curFrameProfiled;
};

#define ENABLE_PROFILE 1
#if ENABLE_PROFILE
    #define PROFILE_ON                          TimerCPU::EnableProfile(true);
    #define PROFILE_OFF                         TimerCPU::EnableProfile(false);
    #define PROFILE_FRAMES(num)                 TimerCPU::SetFramesTobeProfiled(num);
    #define PROFILE_INCREASE                    TimerCPU::IncreaseFrame();
    #define PROFILE_BEGIN(filepath, numFrames)  PROFILE_ON; PROFILE_FRAMES(numFrames); Profile::Get().Begin(filepath);
    #define PROFILE_END                         Profile::Get().End(); PROFILE_OFF
    #define PROFILE_FUNCTION                    TimerCPU timer##__LINE__(__PRETTY_FUNCTION__, true);
    #define PROFILE_SCOPE(name)                 TimerCPU timer##__LINE__(name, true);
#else
    #define PROFILE_ON             
    #define PROFILE_OFF            
    #define PROFILE_FRAMES(num)    
    #define PROFILE_INCREASE       
    #define PROFILE_BEGIN(filepath, numFrames)
    #define PROFILE_END            
    #define PROFILE_FUNCTION       
    #define PROFILE_SCOPE(name)    
#endif

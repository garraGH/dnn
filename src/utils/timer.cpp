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



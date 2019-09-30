/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : /home/garra/study/opengl/timer.h
* author      : Garra
* time        : 2019-09-03 20:36:57
* description : 
*
============================================*/


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

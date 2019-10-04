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
    virtual float GetElapsedTime()  = 0;
    virtual float GetDeltaTime()  = 0;
};

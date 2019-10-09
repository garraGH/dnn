/*============================================
* Copyright(C)2019 Garra. All rights reserved.
* 
* file        : core.h
* author      : Garra
* time        : 2019-09-25 21:22:34
* description : 
*
============================================*/


#pragma once

#include <sys/signal.h>
#include "logger.h"

#define ENABLE_ASSERT
#ifdef ENABLE_ASSERT
    #define ASSERT(x, ...) { if(!(x)) { ERROR("Assertion Failed: {0}", __VA_ARGS__); raise(SIGTRAP); } } 
    #define CORE_ASSERT(x, ...) { if(!(x)) { CORE_ERROR("Assertion Failed: {0}", __VA_ARGS__); raise(SIGTRAP); } }
#else
    #define ASSERT(x, ...)
    #define CORE_ASSERT(x, ...)
#endif

#define BIND_EVENT_CALLBACK(Class, Method) std::bind(&Class::Method, this, std::placeholders::_1)

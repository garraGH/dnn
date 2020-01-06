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

#ifdef _MSC_VER
#define DEBUG_BREAK __debugbreak();
#else
#define DEBUG_BREAK __builtin_trap();
#endif

#define ENABLE_ASSERT
#ifdef ENABLE_ASSERT
    #define ASSERT(x, ...) { if(!(x)) { ERROR("Assertion Failed: {0}", __VA_ARGS__); DEBUG_BREAK; } } 
    #define CORE_ASSERT(x, ...) { if(!(x)) { CORE_ERROR("Assertion Failed: {0}", __VA_ARGS__); DEBUG_BREAK; } }
#else
    #define ASSERT(x, ...)
    #define CORE_ASSERT(x, ...)
#endif

#define STOP ASSERT(false, "stop");

#define BIND_EVENT_CALLBACK(Class, Method) std::bind(&Class::Method, this, std::placeholders::_1)


#ifdef DEBUG
#define GLCheck(x) GLClearError();\
    x;\
    ASSERT(GLLogCall(#x, __FILE__, __LINE__))
#else
#define GLCheck(x) x
#endif

#define COUNT(x) sizeof(x)/sizeof(x[0])

void GLClearError();
bool GLLogCall(const char* function, const char* file, int line);
